"""
文件名: Code/Chapter06/C02_GradClip/main.py
创建时间: 2023/4/9 3:31 下午
"""
import torch


def test_grad_clip(clip_way='value', clip_value=0.8):
    w = torch.tensor([[1.5, 0.5, 3.0], [0.5, 1., 2.]],
                     dtype=torch.float32, requires_grad=True)
    b = torch.tensor([2., 0.5, 3.5], dtype=torch.float32, requires_grad=True)
    x = torch.tensor([[2, 3.]], dtype=torch.float32)
    y = torch.mean(torch.matmul(x, w ** 2) + b ** 2)
    y.backward()
    print("# 梯度裁剪前: ")
    print(f"grad_w: {w.grad}")
    print(f"grad_b: {b.grad}")
    if clip_way == 'value':
        torch.nn.utils.clip_grad_value_([w, b], clip_value)
    else:
        torch.nn.utils.clip_grad_norm_([w, b], max_norm=1.2, norm_type=2.0)
    print(f"# 梯度裁剪后: ")
    print(f"grad_w: {w.grad}")
    print(f"grad_b: {b.grad}")


def clip_grad_norm_(grads, max_norm: float, norm_type: float = 2.0):
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = torch.norm(torch.stack([torch.norm(g, norm_type) for g in grads]), norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for g in grads:
        g.mul_(clip_coef_clamped)
    return total_norm


if __name__ == '__main__':
    test_grad_clip("norm")
