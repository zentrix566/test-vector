"""最小示例：用本地向量模型 all-MiniLM-L6-v2 把文字向量化。

基于 fastembed（底层 ONNX，无需 torch），首次运行会自动下载模型（约 80MB）。
完全离线、免费、不限量，适合本地研究向量数据库。

运行：
    python embed_local.py
"""

from fastembed import TextEmbedding

from config import EN_MODEL, MODEL_DIR


def main() -> None:
    # 首次构造时会下载并缓存模型到 MODEL_DIR
    model = TextEmbedding(model_name=EN_MODEL, cache_dir=MODEL_DIR)

    texts = [
        "Hello, the world of vector databases.",
        "I love studying machine learning.",
        "今天天气真不错。",
    ]

    # embed 返回生成器，每个元素是一个 numpy 向量
    vectors = list(model.embed(texts))

    for text, vec in zip(texts, vectors):
        print(f"文本: {text}")
        print(f"  维度: {len(vec)}  前 5 维: {vec[:5]}")
        print()


if __name__ == "__main__":
    main()
