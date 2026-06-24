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
- 原理演示：用纯 numpy 展示向量距离、模型微调、反向传播的内部机制

## 模型存放位置

所有向量模型统一缓存在 `F:\models`（`config.py` 里的 `MODEL_DIR` 常量），避免散落在系统临时目录。首次运行脚本会自动下载到这里，之后离线复用。当前包含：

- `models--qdrant--all-MiniLM-L6-v2-onnx`（英文，384 维，约 87M）
- `models--Qdrant--bge-small-zh-v1.5`（中文，512 维，约 91M）

如需换目录，修改 `config.py` 里的 `MODEL_DIR` 即可。

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

安装运行所需依赖：

```bash
pip install fastembed chromadb numpy
```

### 3. 运行

本地向量化（首次运行会自动下载模型，约 80MB，之后完全离线）：

```bash
python embed_local.py
```

向量检索（用 Chroma 存文档并查询语义最接近的结果）：

```bash
python chroma_demo.py
```

告警匹配知识库（新告警匹配已知问题，命中给处置办法，未命中自动入库）：

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

```bash
curl -X POST http://127.0.0.1:8000/match -H "Content-Type: application/json" -d "{\"alert\": \"网关大量 502 后端无响应\"}"
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

## 目录结构

- `config.py` —— 共享配置：模型缓存目录、数据库路径、常用模型名
- `embed_local.py` —— 本地 all-MiniLM-L6-v2 向量化脚本
- `chroma_demo.py` —— Chroma 向量数据库语义检索脚本
- `alert_kb.py` —— 告警匹配知识库：匹配已知问题/自动入库（中文 bge 模型）
- `api.py` —— FastAPI Web 服务：POST /match 对外提供告警匹配接口
- `call_api.py` —— 调用 Web 服务的小客户端（测试 /match 接口）
- `static/` —— Swagger UI 本地静态资源（让 /docs 离线可用）
- `verify_model.py` —— 验证中文模型 bge-small-zh：打印路径、维度，做相似度自检
- `explain_vectors.py` —— 原理演示：打印向量、距离、相似度的每一步数字
- `tiny_finetune.py` —— 微调演示：纯 numpy 展示微调的 4 步循环
- `backprop_demo.py` —— 反向传播演示：两层网络手写反向传播解 XOR
- `chroma_db/` —— Chroma 本地持久化数据（不提交）
- `env/` —— 本地虚拟环境（不提交）

## 作者

zentrix566

## 许可证

[MIT](./LICENSE)
