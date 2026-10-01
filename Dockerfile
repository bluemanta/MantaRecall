FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
# 国内构建可传 --build-arg PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple 提速
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --upgrade pip && pip install --index-url "$PIP_INDEX_URL" -r requirements.txt

COPY app ./app
COPY config.yaml ./config.yaml

EXPOSE 8000

# WORKERS 可通过环境变量覆盖
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WORKERS:-2} --log-level ${LOG_LEVEL:-info}"]
