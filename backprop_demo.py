"""反向传播演示：一个两层小神经网络，手写反向传播解 XOR（异或）。

XOR 规律：两个输入相同输出 0，不同输出 1。
  [0,0]->0   [0,1]->1   [1,0]->1   [1,1]->0
单层网络学不会它，必须两层 + 非线性激活，所以是经典教学例子。

本网络结构（共 17 个参数）：
  输入(2) --W1,b1--> 隐藏层(4) --激活--> --W2,b2--> 输出(1)

重点看：反向传播如何从最终误差，反着一层层算出每个参数的调整方向，
       然后一次性更新全部 17 个参数。

运行：
    python backprop_demo.py
"""

import numpy as np

np.random.seed(42)  # 固定随机，结果可复现


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------- 数据：XOR ----------
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)

# ---------- 随机初始化全部参数（一开始模型是个废物）----------
W1 = np.random.randn(2, 4)   # 输入层 -> 隐藏层 的权重（8 个）
b1 = np.zeros((1, 4))        # 隐藏层偏置（4 个）
W2 = np.random.randn(4, 1)   # 隐藏层 -> 输出层 的权重（4 个）
b2 = np.zeros((1, 1))        # 输出偏置（1 个）

lr = 0.5
epochs = 8000


def forward(X):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)         # 隐藏层输出
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)         # 最终预测
    return z1, a1, z2, a2


print("训练前的预测（应接近 0,1,1,0）:")
print(np.round(forward(X)[3].ravel(), 3), "\n")

for epoch in range(1, epochs + 1):
    # ===== 前向 =====
    z1, a1, z2, a2 = forward(X)

    # ===== 损失（均方误差）=====
    loss = np.mean((a2 - Y) ** 2)

    # ===== 反向传播：从输出端反着往回算每个参数的梯度 =====
    # 输出层：误差 × sigmoid 的导数
    d_a2 = 2 * (a2 - Y) / len(X)
    d_z2 = d_a2 * a2 * (1 - a2)
    d_W2 = a1.T @ d_z2          # 输出层权重该怎么调
    d_b2 = d_z2.sum(axis=0, keepdims=True)

    # 把误差“传回”隐藏层（这一步就是反向传播的精髓）
    d_a1 = d_z2 @ W2.T
    d_z1 = d_a1 * a1 * (1 - a1)
    d_W1 = X.T @ d_z1          # 隐藏层权重该怎么调
    d_b1 = d_z1.sum(axis=0, keepdims=True)

    # ===== 一次性更新全部 17 个参数 =====
    W2 -= lr * d_W2
    b2 -= lr * d_b2
    W1 -= lr * d_W1
    b1 -= lr * d_b1

    if epoch % 1000 == 0:
        print(f"第{epoch:>4}轮  loss={loss:.5f}")

pred = forward(X)[3].ravel()
print("\n训练后的预测（目标 0,1,1,0）:")
print(np.round(pred, 3))
print("四舍五入:", np.round(pred).astype(int), "  全对说明网络学会了 XOR")
print("\n反向传播把误差从输出反推回每一层，自动更新了全部 17 个参数。")
