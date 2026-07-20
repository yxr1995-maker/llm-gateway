# Run persistently on macOS with launchd

Register the gateway as a launchd user agent for auto-start on login + crash recovery.

```bash
# 1. clone to a non-TCC-protected dir (don't use ~/Documents, ~/Desktop, ~/Downloads;
#    otherwise launchd execution is blocked with Operation not permitted)
git clone <repo-url> ~/llm-gateway && cd ~/llm-gateway

# 2. prepare env and config
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp config.example.yaml config.yaml   # edit and fill in keys

# 3. replace /Users/YOUR_NAME in the plist with your real path
sed -i '' "s|/Users/YOUR_NAME|$HOME|g" contrib/macos/com.llm-gateway.plist

# 4. install and start
mkdir -p data
cp contrib/macos/com.llm-gateway.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.llm-gateway.plist

# verify
launchctl list | grep llm-gateway
curl http://127.0.0.1:8080/v1/models -H "Authorization: Bearer <master_key>"
```

Common ops:

```bash
kill $(lsof -tnP -iTCP:8080 -sTCP:LISTEN)   # restart (KeepAlive auto-respawns)
launchctl unload ~/Library/LaunchAgents/com.llm-gateway.plist   # stop
tail -f data/gateway.err.log                # logs
```
