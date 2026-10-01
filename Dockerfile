FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
# 国内构建可传 --build-arg PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple 提速
ARG PIP_INDEX_URL=https://pypi.org/simple
# torch 用 CPU-only 版本：reranker 只需 CPU 推理，省 ~2GB 磁盘与内存
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# HuggingFace 镜像：国内构建下载 reranker 模型用
ARG HF_MIRROR=https://hf-mirror.com
RUN pip install --upgrade pip \
    && pip install --index-url "$TORCH_INDEX_URL" torch \
    && pip install --index-url "$PIP_INDEX_URL" -r requirements.txt

# 把 cross-encoder reranker 模型 baked 进镜像，运行时不依赖外网
# （~90MB；sentence-transformers 已随 requirements 装好）
ENV HF_ENDPOINT=${HF_MIRROR}
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('cross-encoder/ms-marco-MiniLM-L-6-v2', local_dir='/app/models/ms-marco-MiniLM-L-6-v2')"

COPY app ./app
COPY config.yaml ./config.yaml

# reranker 默认配置（可用 .env 中的 RERANKER_MODEL / RERANK_TOP_N 覆盖）
ENV RERANKER_MODEL=/app/models/ms-marco-MiniLM-L-6-v2 \
    RERANK_TOP_N=50

EXPOSE 8000

# WORKERS 可通过环境变量覆盖
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WORKERS:-2} --log-level ${LOG_LEVEL:-info}"]
