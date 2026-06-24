"""最小微调演示：用纯 numpy 展示"微调"的灵魂——那 4 步循环。

真实的大模型/语音/视频模型微调，本质都是同样的循环，只是模型更大、
数据更多。这里用一个只有 2 个权重的玩具模型，让你看清每一步。

场景设定：
  - "预训练"模型：已经学到 y = 1*x + 0（在旧数据上）
  - 现在拿到新领域的数据，真实规律其实是 y ≈ 2*x + 1
  - 我们"微调"模型，让它适应新数据

运行：
    python tiny_finetune.py
"""

import numpy as np

# ---------- "预训练"得到的初始权重（微调的起点，不是从零随机）----------
w = 1.0
b = 0.0

# ---------- 新领域的训练数据（输入 x，正确答案 y）----------
x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([3.0, 5.0, 7.0, 9.0])   # 真实规律 y = 2x + 1

lr = 0.01      # 学习率：每次权重挪多大一步
epochs = 200   # 把数据反复看 200 遍


def predict(x):
    return w * x + b


print(f"微调前: w={w:.3f}, b={b:.3f}")
print(f"  对 x=5 的预测 = {predict(5):.3f}   (正确答案应是 11)\n")

for epoch in range(1, epochs + 1):
    # ① 前向：算预测
    pred = predict(x)

    # ② 算损失：预测和正确答案的平均平方误差（MSE）
    loss = np.mean((pred - y) ** 2)

    # ③ 算梯度：loss 对每个权重的偏导（指明该往哪调）
    grad_w = np.mean(2 * (pred - y) * x)
    grad_b = np.mean(2 * (pred - y))

    # ④ 更新权重：朝让 loss 变小的方向挪一小步
    w -= lr * grad_w
    b -= lr * grad_b

    if epoch % 40 == 0:
        print(f"第{epoch:>3}轮  loss={loss:7.4f}  w={w:.3f}  b={b:.3f}")

print(f"\n微调后: w={w:.3f}, b={b:.3f}   (目标 w=2, b=1)")
print(f"  对 x=5 的预测 = {predict(5):.3f}   (正确答案是 11)")
print("\n看：loss 一路下降，w/b 被'调'到了贴合新数据的值——这就是微调。")
