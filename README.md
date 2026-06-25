# test-vector

一个研究向量数据库的最小实验项目，全部基于本地模型，离线、免费、不限量：用 `all-MiniLM-L6-v2` 把文字向量化，再用 Chroma 做语义检索，并附带原理演示脚本。

## 学习路线图

本项目从"会用"一路打通到"懂原理"，建议按下面顺序阅读和运行脚本：

| 顺序 | 脚本 | 学到什么 |
|---|---|---|
| 1️⃣ | `embed_local.py` | 文字怎么变成向量（本地模型，384 维）|
| 2️⃣ | `chroma_demo.py` | 用 Chroma 向量数据库做语义检索 |
| 3️⃣ | `explain_vectors.py` | 检索原理：向量距离 / 余弦相似度怎么算 |
| 4️⃣ | `tiny_finetune.py` | 模型微调的 4 步循环（前向→损失→梯度→更新）|
| 5️⃣ | `backprop_demo.py` | 反向传播如何自动更新一个网络的全部参数 |

## 主要功能

- 向量化：基于 fastembed（底层 ONNX，无需 torch），用 `all-MiniLM-L6-v2` 离线生成 384 维向量
- 检索：用 Chroma 向量数据库存入多段文字，输入查询返回语义最接近的若干条（默认模型同为 all-MiniLM-L6-v2）
- 告警匹配：新告警在知识库里找最相似的已知问题，命中给出根因和处置办法，未命中自动入库（`alert_kb.py`，中文 bge 模型）
- RAG 生成：在检索命中的基础上，把"已知问题+根因+处置办法"作为上下文喂给大模型，针对当前告警生成定制化处置建议（`rag.py`，OpenAI 兼容接口；未配密钥时自动降级返回知识库原文）
- 原理演示：用纯 numpy 展示向量距离、模型微调、反向传播的内部机制

## 模型存放位置

所有向量模型统一缓存在 `MODEL_DIR` 指定的目录（默认 `E:\models`），避免散落在系统临时目录。首次运行脚本会自动下载到这里，之后离线复用。当前包含：

- `models--qdrant--all-MiniLM-L6-v2-onnx`（英文，384 维，约 87M）
- `models--Qdrant--bge-small-zh-v1.5`（中文，512 维，约 91M）

路径按机器而异，已做成可配置项：换电脑时把 `.env` 里的 `MODEL_DIR` 改成本机目录即可（不配则用默认 `E:\models`）。

### 模型怎么下载到 E 盘

**不需要手动下载**。代码用 `fastembed` 加载模型时传了 `cache_dir=MODEL_DIR`（见 `config.py`），首次构造模型对象就会自动把模型下载并缓存到该目录：

```python
# embed_local.py / alert_kb.py 里的写法
model = TextEmbedding(model_name=EN_MODEL, cache_dir=MODEL_DIR)
```

所以下载到 E 盘只需两步：

1. 配好目录（默认就是 `E:\models`，要换盘就改 `.env` 里的 `MODEL_DIR`，目录不存在会自动创建）：

   ```bash
   copy .env.example .env
   ```

2. 运行下载脚本，一次性把项目用到的全部模型拉到 `MODEL_DIR`（需联网，约 80～180MB，之后完全离线复用）：

   ```bash
   python download_models.py
   ```

   也可以不专门下载，直接跑任意用到模型的脚本（如 `python embed_local.py`），首次运行会自动下载用到的那个模型。

下载完成后，`E:\models` 下会出现 `models--qdrant--all-MiniLM-L6-v2-onnx`、`models--Qdrant--bge-small-zh-v1.5` 等目录，再次运行直接从本地读取，不再联网。

> 注意：只有**向量模型**需要这样下载。Chroma 向量数据库（`chromadb`）是个 Python 库，在 `pip install -r requirements.txt` 那步就装好了，**不需要单独下载**；它的数据运行时写到本地 `chroma_db/` 目录。

## 运行方式

### 1. 创建并激活虚拟环境

创建本地虚拟环境：

```bash
python -m venv env
```

激活虚拟环境（Windows CMD）：

```bash
env\Scripts\activate
```

### 2. 安装依赖

一键安装运行所需的全部依赖（版本已在 `requirements.txt` 中锁定）：

```bash
pip install -r requirements.txt
```

### 3. 配置大模型（仅 RAG 生成功能需要）

向量检索、告警匹配全程本地离线，无需任何密钥。只有"RAG 生成处置建议"这步要调用大模型。复制示例配置并填入自己的 key：

```bash
copy .env.example .env
```

然后编辑 `.env`，填入 `OPENAI_API_KEY`，按需修改 `OPENAI_BASE_URL` 和 `LLM_MODEL`（默认对接 DeepSeek，改一行 base_url 即可换通义/OpenAI 等任意 OpenAI 兼容厂商）。**不配密钥也能跑**：RAG 接口会自动降级，直接返回知识库里的原始处置办法。

### 4. 运行

本地向量化（首次运行会自动下载模型，约 80MB，之后完全离线）：

```bash
python embed_local.py
```

向量检索（用 Chroma 存文档并查询语义最接近的结果）：

```bash
python chroma_demo.py
```

告警匹配知识库（新告警匹配已知问题，命中给处置办法，未命中自动入库；末尾附带一段 RAG 生成演示）：

```bash
python alert_kb.py
```

验证中文模型（打印模型路径、向量维度，做中文相似度自检）：

```bash
python verify_model.py
```

启动告警匹配 Web 服务（FastAPI，把模型加载进内存并对外提供接口）：

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

启动后访问交互式文档 http://127.0.0.1:8000/docs （Swagger UI 资源已本地化到 `static/`，国内可秒开），或直接调用接口：

仅检索匹配（不调用大模型）：

```bash
curl -X POST http://127.0.0.1:8000/match -H "Content-Type: application/json" -d "{\"alert\": \"网关大量 502 后端无响应\"}"
```

检索 + RAG 生成处置建议（需在 `.env` 配好大模型密钥；未配则降级返回知识库原文）：

```bash
curl -X POST http://127.0.0.1:8000/advise -H "Content-Type: application/json" -d "{\"alert\": \"网关大量 502 后端无响应\"}"
```

原理演示（打印向量、距离、相似度的每一步数字）：

```bash
python explain_vectors.py
```

微调演示（纯 numpy 展示微调的 4 步循环，看 loss 下降、权重被调动）：

```bash
python tiny_finetune.py
```

反向传播演示（两层网络手写反向传播解 XOR，看误差如何反向更新全部参数）：

```bash
python backprop_demo.py
```

## 常用命令

激活虚拟环境（每次新开终端都要先激活）：

```bash
env\Scripts\activate
```

运行本地向量化脚本：

```bash
python embed_local.py
```

运行 Chroma 检索脚本：

```bash
python chroma_demo.py
```

## 查看向量库里存了什么

Chroma 的数据在 `chroma_db/` 目录，分两部分存：**文本/metadata 在 `chroma.sqlite3`**，**向量数值在子目录的 `.bin` 索引文件里**（二进制，读不了）。所以分两种方式看：

### 方式一：看原文和 metadata（SQLite 客户端）

用 DBeaver、DB Browser for SQLite 等工具打开 `chroma_db/chroma.sqlite3`。表很多，但真正存你数据的就 3 张，其余都是 Chroma 内部用的（索引、写日志、多租户、迁移记录等），不用管：

- `collections` —— 你建的集合（名字、id、维度、距离配置）
- `embeddings` —— 每条记录一行，内部 `id` 对应你的 `embedding_id`（如 doc1）
- `embedding_metadata` —— 文档原文 + metadata，**原文存在 `key='chroma:document'` 的 `string_value` 里**

把记录 id 和原文拼到一起看，在客户端里跑这条 SQL：

```sql
SELECT e.embedding_id AS doc_id,
       m.string_value  AS document
FROM embeddings e
JOIN embedding_metadata m ON e.id = m.id
WHERE m.key = 'chroma:document'
ORDER BY e.embedding_id;
```

去掉 `WHERE` 就能看到 `root_cause`、`service` 等其它 metadata 字段（它们也在 `embedding_metadata`，只是 `key` 不同）。

### 方式二：看向量数值（脚本）

向量不在 sqlite 里，得走 Chroma 的 API 取。直接跑：

```bash
python inspect_chroma.py
```

会列出所有集合，并打印每条的原文、metadata 和向量预览（维度 + 前几维）。

### 方式三：图形客户端（可选）

Chroma server 是纯 API、没有自带网页，所以浏览器开 `http://localhost:8000/` 是 404 属正常。先把库跑成服务：

```bash
chroma run --path ./chroma_db --port 8000
```

想在浏览器里快速看集合列表，可直接访问接口（返回 JSON）：

```
http://localhost:8000/api/v2/tenants/default_tenant/databases/default_database/collections
```

要真正的可视化界面，可另起开源的 chromadb-admin 等前端连这个服务（注意第三方 UI 对 Chroma v2 API 的兼容性可能滞后，连不上时用方式一/二最稳）。

## 目录结构

- `config.py` —— 共享配置：模型缓存目录、数据库路径、常用模型名、大模型接口配置（从 `.env` 读密钥）
- `download_models.py` —— 一键把项目用到的全部向量模型下载到 `MODEL_DIR`
- `inspect_chroma.py` —— 查看 Chroma 本地库存了什么（列出集合、打印每条的原文/metadata/向量预览）
- `inspect_model.py` —— 分词器 + ONNX 模型二合一查看工具：看清 文字 → token id → 向量 的完整链路，支持 `all/tokenizer/onnx` 三种模式，默认识别 `MODEL_DIR` 也可命令行指定路径
- `embed_online.py` —— 在线向量模型调用：通过火山方舟 Embedding API 把文字转成向量，与本地模型互补，中文效果通常更好
- `embed_local.py` —— 本地 all-MiniLM-L6-v2 向量化脚本
- `chroma_demo.py` —— Chroma 向量数据库语义检索脚本
- `alert_kb.py` —— 告警匹配知识库：匹配已知问题/自动入库（中文 bge 模型），含 `match_and_advise` 完整 RAG 入口
- `rag.py` —— RAG 生成环节：把检索结果作为上下文，调用大模型生成定制化处置建议
- `api.py` —— FastAPI Web 服务：POST /match 仅检索；POST /advise 检索+生成（完整 RAG）
- `call_api.py` —— 调用 Web 服务的小客户端（测试 /match 接口）
- `static/` —— Swagger UI 本地静态资源（让 /docs 离线可用）
- `verify_model.py` —— 验证中文模型 bge-small-zh：打印路径、维度，做相似度自检
- `explain_vectors.py` —— 原理演示：打印向量、距离、相似度的每一步数字
- `tiny_finetune.py` —— 微调演示：纯 numpy 展示微调的 4 步循环
- `backprop_demo.py` —— 反向传播演示：两层网络手写反向传播解 XOR
- `requirements.txt` —— 项目依赖清单（`pip install -r requirements.txt` 一键安装）
- `.env.example` —— 大模型接口配置示例（复制为 `.env` 填入真实密钥，`.env` 不提交）
- `chroma_db/` —— Chroma 本地持久化数据（不提交）：`chroma.sqlite3` 存原文/id/metadata，子目录里的 `.bin` 是向量索引（二进制）。想看里面存了什么用 `python inspect_chroma.py`
- `env/` —— 本地虚拟环境（不提交）

## 作者

zentrix566

## 许可证

[MIT](./LICENSE)
