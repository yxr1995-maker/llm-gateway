# llm-gateway 镜像
# 构建：docker build -t llm-gateway .
# 运行（挂载配置文件与数据目录，详见 README.md）：
#   docker run -d --name llm-gateway \
#     -p 8080:8080 \
#     -v $(pwd)/config.yaml:/app/config.yaml \
#     -v $(pwd)/data:/app/data \
#     llm-gateway
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖，充分利用构建缓存
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝应用代码与静态文件
COPY app ./app
COPY static ./static

# 运行时通过 -v 挂载：
#   /app/config.yaml   配置文件（必需，由 config.example.yaml 复制修改而来）
#   /app/data          用量统计 SQLite 数据目录（可选，不挂载则数据随容器生命周期）
EXPOSE 8080

CMD ["python", "-m", "app.main"]
