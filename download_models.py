"""一键把项目用到的向量模型下载到本地 MODEL_DIR。

平时各脚本首次运行会自动下载模型，但那是用到哪个下哪个。
本脚本把全部模型一次性拉齐，方便在新机器上提前备好、之后完全离线运行。

模型缓存目录由 config.MODEL_DIR 决定（默认 E:\\models，可在 .env 改 MODEL_DIR）。

运行：
    python download_models.py
"""

from fastembed import TextEmbedding

from config import EN_MODEL, ZH_MODEL, MODEL_DIR

# 需要预下载的模型清单：(模型名, 中文说明)
MODELS = [
    (EN_MODEL, "英文，384 维"),
    (ZH_MODEL, "中文，512 维"),
]


def download(model_name: str) -> None:
    """下载并缓存单个模型，用一次推理确认模型确实可用。"""
    # 构造对象时 fastembed 会把模型下载到 cache_dir；embed 一条短文本触发实际加载
    model = TextEmbedding(model_name=model_name, cache_dir=MODEL_DIR)
    vec = next(iter(model.embed(["test"])))
    print(f"  完成，向量维度 {len(vec)}")


def main() -> None:
    print(f"模型缓存目录：{MODEL_DIR}\n")
    for name, desc in MODELS:
        print(f"下载 {name}（{desc}）...")
        download(name)
        print()
    print("全部模型已就绪，之后可完全离线运行。")


if __name__ == "__main__":
    main()
