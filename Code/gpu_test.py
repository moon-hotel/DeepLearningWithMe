def gpu_test():
    import torch
    device = torch.device('cuda:0')
    a = torch.randn([2, 3, 4]).to(device)
    b = torch.randn([2, 3, 4]).to(device)
    c = a + b
    print(c.device)
    print("PyTorch版本:", torch.__version__)
    print("GPU是否可用:", torch.cuda.is_available())
    print("GPU数量:", torch.cuda.device_count())  # 查看GPU个数
    print("CUDA版本:", torch.version.cuda)  # )
    print("cuDNN是否启用:", torch.backends.cudnn.enabled)

if __name__ == '__main__':
    gpu_test()
    # cuda:0
    # PyTorch版本: 2.0.1+cu118
    # GPU是否可用: True
    # GPU数量: 1
    # CUDA版本: 11.8
    # cuDNN是否启用: True
