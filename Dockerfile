FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
# 国内构建可传 --build-arg PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple 提速
ARG PIP_INDEX_URL=https://pypi.org/simple
# torch CPU-only wheel：pytorch.org 从国内直连极慢（~50kB/s），默认走阿里云镜像；
# 可用 --build-arg TORCH_WHL_URL=... 覆盖。reranker 只需 CPU 推理。
ARG TORCH_WHL_URL=https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.9.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl
RUN pip install --upgrade pip \
    && pip install "$TORCH_WHL_URL" \
    && pip install --index-url "$PIP_INDEX_URL" -r requirements.txt

# 把 cross-encoder reranker 模型 baked 进镜像，运行时不依赖外网（~90MB）。
# 注意：huggingface.co / hf-mirror.com 从该服务器网络不可达，改走 ModelScope。
# 模型真实 ID 为 cross-encoder/ms-marco-MiniLM-L6-v2（L6 中间无连字符）。
RUN pip install --index-url "$PIP_INDEX_URL" modelscope \
    && python -c "from modelscope import snapshot_download; snapshot_download('cross-encoder/ms-marco-MiniLM-L6-v2', local_dir='/app/models/ms-marco-MiniLM-L6-v2')"

COPY app ./app
COPY config.yaml ./config.yaml

# reranker 默认配置（可用 .env 中的 RERANKER_MODEL / RERANK_TOP_N 覆盖）
ENV RERANKER_MODEL=/app/models/ms-marco-MiniLM-L6-v2 \
    RERANK_TOP_N=50

EXPOSE 8000

# WORKERS 可通过环境变量覆盖
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WORKERS:-2} --log-level ${LOG_LEVEL:-info}"]
