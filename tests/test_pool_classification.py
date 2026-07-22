"""Unit tests for KeyPool failure classification (opencodex-style policy).

Run: .venv/bin/python -m pytest tests/test_pool_classification.py
     .venv/bin/python tests/test_pool_classification.py   (also works standalone)
"""
from __future__ import annotations
import time
from app.pool import (KeyPool, CREDENTIAL_COOLDOWN, QUOTA_COOLDOWN_MIN,
                      QUOTA_COOLDOWN_MAX, TRANSIENT_LADDER)

_results: list[tuple[str, bool, str]] = []

def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, cond, detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"  -> {detail}"))


def test_credential_long_cooldown_and_reauth():
    p = KeyPool(); p.register("prov", ["k1", "k2"])
    out = p.report_failure("prov", "k1", status_code=401)
    check("401 -> credential", out == "credential", out)
    check("401 marks needs_reauth", p.needs_reauth("prov") == ["k1"], p.needs_reauth("prov"))
    # k1 must be cooled (not acquirable), k2 still free
    check("401 cools the key", p.acquire("prov") == "k2", "k1 should be skipped")
    check("401 cooldown ~1h", abs(p._cooldown[("prov","k1")] - (time.monotonic()+CREDENTIAL_COOLDOWN)) < 5)


def test_quota_respects_retry_after():
    p = KeyPool(); p.register("prov", ["k1", "k2"])
    out = p.report_failure("prov", "k1", status_code=429, retry_after=120)
    check("429 -> quota", out == "quota", out)
    check("429 cools the key", p.acquire("prov") == "k2")
    check("429 retry_after honored", abs(p._cooldown[("prov","k1")] - (time.monotonic()+120)) < 5)
    # no retry-after -> floor
    out2 = p.report_failure("prov", "k2", status_code=429)
    check("429 no retry_after -> floor", abs(p._cooldown[("prov","k2")] - (time.monotonic()+QUOTA_COOLDOWN_MIN)) < 5)


def test_quota_capped():
    p = KeyPool(); p.register("prov", ["k1"])
    p.report_failure("prov", "k1", status_code=429, retry_after=10**9)
    check("429 retry_after capped to 24h",
          abs(p._cooldown[("prov","k1")] - (time.monotonic()+QUOTA_COOLDOWN_MAX)) < 5)


def test_caller_error_does_not_cool_key():
    """400/404/422 must NOT punish the key (the request is bad, not the key)."""
    p = KeyPool(); p.register("prov", ["k1", "k2"])
    out = p.report_failure("prov", "k1", status_code=400)
    check("400 -> caller", out == "caller", out)
    # k1 must STILL be acquirable (round-robin continues from where it left)
    avail = p.available("prov")
    check("400 leaves both keys available", avail == 2, f"available={avail}")
    check("400 not in cooldown", ("prov", "k1") not in p._cooldown)
    check("404 -> caller too", p.report_failure("prov", "k1", status_code=404) == "caller")


def test_transient_escalating_backoff():
    p = KeyPool(); p.register("prov", ["k1"])
    expected = []
    for i, want in enumerate(TRANSIENT_LADDER):
        out = p.report_failure("prov", "k1", status_code=500)
        check(f"500#{i} -> transient", out == "transient", out)
        expected.append(want)
        check(f"500#{i} backoff={want}s",
              abs(p._cooldown[("prov","k1")] - (time.monotonic()+want)) < 5,
              f"got ladder step {i}")
    # further failures stay capped at the last rung
    p.report_failure("prov", "k1", status_code=503)
    check("500 capped at last rung",
          abs(p._cooldown[("prov","k1")] - (time.monotonic()+TRANSIENT_LADDER[-1])) < 5)


def test_network_error_is_transient():
    """status_code=None (network/timeout/unknown) -> transient."""
    p = KeyPool(); p.register("prov", ["k1"])
    out = p.report_failure("prov", "k1")  # no status_code
    check("None -> transient", out == "transient", out)


def test_success_resets_everything():
    p = KeyPool(); p.register("prov", ["k1"])
    p.report_failure("prov", "k1", status_code=401)   # sets cooldown + reauth + (no fail_count)
    p.report_failure("prov", "k1", status_code=500)   # sets fail_count
    p.report_success("prov", "k1")
    check("success clears cooldown", ("prov", "k1") not in p._cooldown)
    check("success clears fail_count", ("prov", "k1") not in p._fail_count)
    check("success clears reauth", p.needs_reauth("prov") == [])


def test_sync_drops_state_for_removed_keys():
    p = KeyPool(); p.register("prov", ["k1", "k2"])
    p.report_failure("prov", "k2", status_code=401)
    p.register("prov", ["k1"])  # k2 removed
    check("removed key cooldown dropped", ("prov", "k2") not in p._cooldown)
    check("removed key reauth dropped", p.needs_reauth("prov") == [])


def main():
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]:
        fn()
    failed = [r for r in _results if not r[1]]
    print(f"\n=== {len(_results)-len(failed)}/{len(_results)} passed ===")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
