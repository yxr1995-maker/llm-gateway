#!/bin/bash
# llm-gateway launch script (for launchd)
# assume the repo is cloned at ~/llm-gateway with a .venv created
set -e
cd "$(dirname "$0")/../.."
exec .venv/bin/python -m app.main
