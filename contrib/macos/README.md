# macOS launchd 常驻运行

把网关注册为 launchd 用户服务，开机自启 + 崩溃自动拉起。

```bash
# 1. 克隆到非 TCC 保护目录（不要放 ~/Documents、~/Desktop、~/Downloads，
#    否则 launchd 执行会被系统拦截，报 Operation not permitted）
git clone <repo-url> ~/llm-gateway && cd ~/llm-gateway

# 2. 准备环境与配置
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp config.example.yaml config.yaml   # 编辑填入 key

# 3. 替换 plist 里的 /Users/YOUR_NAME 为你的实际路径
sed -i '' "s|/Users/YOUR_NAME|$HOME|g" contrib/macos/com.llm-gateway.plist

# 4. 安装并启动
mkdir -p data
cp contrib/macos/com.llm-gateway.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.llm-gateway.plist

# 验证
launchctl list | grep llm-gateway
curl http://127.0.0.1:8080/v1/models -H "Authorization: Bearer <master_key>"
```

常用运维：

```bash
kill $(lsof -tnP -iTCP:8080 -sTCP:LISTEN)   # 重启（KeepAlive 自动拉起）
launchctl unload ~/Library/LaunchAgents/com.llm-gateway.plist   # 停止
tail -f data/gateway.err.log                # 日志
```
