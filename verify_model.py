"""验证中文向量模型 bge-small-zh-v1.5 是否正常工作。

打印模型信息、向量维度，并做一个中文相似度自检：
意思相近的中文句子相似度应明显高于无关句子。

运行：
    python verify_model.py
"""

import numpy as np
from fastembed import TextEmbedding

from config import ZH_MODEL, MODEL_DIR


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    model = TextEmbedding(model_name=ZH_MODEL, cache_dir=MODEL_DIR)

    # fastembed 会把模型下载到这个本地缓存目录
    print(f"模型: {ZH_MODEL}")
    print(f"缓存目录: {model.model._model_dir}\n")

    base = "订单服务连接池耗尽，请求超时"
    similar = "order 服务连接池满了，大量请求超时"   # 含义相近
    unrelated = "今天北京天气晴朗适合出游"            # 完全无关

    vectors = list(model.embed([base, similar, unrelated]))
    v_base, v_similar, v_unrelated = vectors

    print(f"向量维度: {len(v_base)}\n")
    print(f"基准句: {base}")
    print(f"  vs 相近句『{similar}』  相似度={cosine(v_base, v_similar):.4f}")
    print(f"  vs 无关句『{unrelated}』  相似度={cosine(v_base, v_unrelated):.4f}")

    if cosine(v_base, v_similar) > cosine(v_base, v_unrelated):
        print("\n✅ 验证通过：相近中文句的相似度明显更高，模型工作正常。")
    else:
        print("\n❌ 异常：相近句相似度没有更高，请检查模型。")


if __name__ == "__main__":
    main()
