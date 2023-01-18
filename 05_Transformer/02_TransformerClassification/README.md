
## A Transformer Framework Based Classification Task

### 一个基于Transformer Encoder网络结构的文本分类模型



## 1. 环境准备
* Python==3.6
* PyTorch==1.5.0
* torchtext==0.6.0


## 2. 使用方法
* STEP 1. 直接下载或克隆本项目
* STEP 2. 可自定义修改配置文件`config.py`中的配置参数，也可以保持默认
### 训练
直接执行如下命令即可进行模型训练：
```
python train.py
```
训练过程：
```python
    Epoch: 9, Batch: [410/469], Train loss 0.186, Train acc: 0.938
    Epoch: 9, Batch: [420/469], Train loss 0.150, Train acc: 0.938
    Epoch: 9, Batch: [430/469], Train loss 0.269, Train acc: 0.941
    Epoch: 9, Batch: [440/469], Train loss 0.197, Train acc: 0.925
    Epoch: 9, Batch: [450/469], Train loss 0.245, Train acc: 0.917
    Epoch: 9, Batch: [460/469], Train loss 0.272, Train acc: 0.902
    Accuracy on test 0.886
```

