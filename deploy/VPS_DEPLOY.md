# 公网 VPS 部署说明（AML 第二期：自部署 Add/Search API）

第二期要求参赛者**自行部署公网可达的 Add/Search API**，平台不再代部署。
评测后**不要求**继续开放接口，用完即下线即可。

## 1. 机器选型

- 配置：**2c4g 起步**（4c8g 更稳，72 小时 Full 的 Search 256 并发主要吃连接数和内存）
- 系统：Ubuntu 22.04 / 24.04（64 位）
- 带宽：5M 起步；Add 阶段是写入密集，Search 阶段是读密集
- embedding / LLM 都走远端 API（DashScope / OpenAI），VPS 本身不跑模型
- 厂商不限：阿里云/腾讯云/华为云按量付费均可，用完释放

> 备选（本骨架已注明）：如果 pgvector 部署遇到困难，可退化为 **SQLite + FTS5**
> 方案——把 `app/db.py` 换成 SQLite 实现、检索改纯词法+本地向量。
> 但 SQLite 扛 256 并发写入较吃力，仅建议作为本机开发备选，正式评测仍推荐 pgvector。

## 2. 基础环境

```bash
# Docker（Ubuntu 24.04）
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER   # 重新登录生效
docker --version && docker compose version
```

## 3. 域名与 DNS（HTTPS 必备）

1. 准备一个域名（或子域名），如 `aml-api.example.com`
2. DNS 加 A 记录指向 VPS 公网 IP
3. 等待解析生效：`dig +short aml-api.example.com` 能返回你的 IP

> 官方要求公网 HTTPS。不要直接把 8000 端口裸露到公网。

## 4. 部署服务

```bash
# 把 ~/workspace/aml 传到服务器（任选其一）
scp -r ~/workspace/aml user@your-vps:/opt/aml
# 或：git push 到私有仓库后在服务器 git clone

cd /opt/aml
cp .env.example .env
vim .env   # 必填：MEMORY_API_KEY（长随机串）、EMBEDDING_API_KEY、LLM_API_KEY
           # 正式评测：EMBEDDING_PROVIDER=openai_compatible，extraction.enabled 按决策来

docker compose up -d --build
docker compose ps
curl http://127.0.0.1:8000/health   # {"status":"ok"}
```

## 5. Caddy 反向代理（自动 HTTPS）

```bash
sudo apt-get install -y caddy
sudo tee /etc/caddy/Caddyfile > /dev/null <<'EOF'
aml-api.example.com {
    reverse_proxy 127.0.0.1:8000
}
EOF
sudo systemctl reload caddy
```

防火墙放行：

```bash
sudo ufw allow 80,443/tcp && sudo ufw enable
# 或云厂商安全组放行 80/443
```

公网验证（在**本地电脑**上执行）：

```bash
curl https://aml-api.example.com/health
curl -X POST https://aml-api.example.com/search \
  -H "X-Api-Key: $MEMORY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query":"ping","user_id":"nobody","top_k":1}'
# 期望：{"data":[]} 且 200
```

再跑一遍公网 Smoke（把 `--base-url` 换成 https 域名）：

```bash
python scripts/smoke.py --base-url https://aml-api.example.com --api-key "$MEMORY_API_KEY"
```

## 6. 提交评测

1. 在官网 Evaluation 页申请评测 Key（申请次日 19:00 前发放，**最晚 10/2 申请**）
2. 绑定你的 API 地址与 Key，跑官方 **Smoke**（验证同步 Add/Search 链路）
3. Smoke 通过后再启动 **Full**（约 72 小时；每个 Key 每赛道最多 2 次 Full，第 2 次须在首次 Full 完成满 30 天后发起——**10/5 是首次 Full 临界点**）
4. 评测期间不要改配置、不要重启丢数据；建议开着 `docker compose logs -f api` 观察

## 7. 评测后下线

```bash
cd /opt/aml
docker compose down -v   # -v 会删掉 pg 数据卷；如需保留审计日志，先备份再删
# 然后在云控制台释放 VPS（停止计费）
```

合规提醒：评测数据与衍生数据只用于当次运行，30 天内删除；不要保留评测原文副本。

## 8. 故障排查

| 现象 | 查法 |
|---|---|
| `/health` 503 | `docker compose logs api` 看 DB 连接；`docker compose ps` 看 db 是否 healthy |
| 启动报 embedding 身份不一致 | 换了模型/维度，必须换新库：`docker compose down -v` 后重起（会清空数据） |
| Add 409 | 正常：相同 request_id 被不同内容复用；检查客户端 request_id 生成逻辑 |
| Search 慢 | 先看 `scripts/load_test.py` 的 p95；调大 `WORKERS`、DB 升配；检查 HNSW 索引是否生效（`EXPLAIN`） |
| 429 / LLM 超时 | DashScope/OpenAI 配额或限流；`config.yaml` 里降 `add_llm_concurrency`，或升配额 |
