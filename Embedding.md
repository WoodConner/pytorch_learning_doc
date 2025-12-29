# Embedding

*class* torch.nn.Embedding(*num_embeddings*, *embedding_dim*, *padding_idx=None*, *max_norm=None*, *norm_type=2.0*, *scale_grad_by_freq=False*, *sparse=False*, *_weight=None*, *_freeze=False*, *device=None*, *dtype=None*)[[source]](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/nn/modules/sparse.py#L15)

一个简单的查找表，存储固定词汇表和大小的嵌入。

此模块通常用于存储词嵌入并通过索引检索它们。模块的输入是索引列表，输出是对应的词嵌入。

参数

- **num_embeddings** ([*int*](https://docs.pythonlang.cn/3/library/functions.html#int "(in Python v3.14)")) – 嵌入字典的大小。

- **embedding_dim** ([*int*](https://docs.pythonlang.cn/3/library/functions.html#int "(in Python v3.14)")) – 每个嵌入向量的大小。

- **padding_idx** ([*int*](https://docs.pythonlang.cn/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) – 如果指定，`padding_idx`处的条目不计入梯度；因此，`padding_idx`处的嵌入向量在训练期间不会被更新，即它保持为一个固定的“pad”。对于新构造的 Embedding，`padding_idx`处的嵌入向量将默认为全零，但可以更新为另一个值以用作填充向量。

- **max_norm** ([*float*](https://docs.pythonlang.cn/3/library/functions.html#float "(in Python v3.14)")*,* *optional*) – 如果给定，则范数大于 `max_norm` 的每个嵌入向量将被重新归一化为范数为 `max_norm`。

- **norm_type** ([*float*](https://docs.pythonlang.cn/3/library/functions.html#float "(in Python v3.14)")*,* *optional*) – 计算 `max_norm` 选项的 p-范数的 p 值。默认为 `2.0`。

- **scale_grad_by_freq** ([*bool*](https://docs.pythonlang.cn/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) – 如果给定，这将通过小批量中词语频率的倒数来缩放梯度。默认值为 `False`。

- **sparse** ([*bool*](https://docs.pythonlang.cn/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) – 如果为 `True`，则`weight`矩阵的梯度将是一个稀疏张量。有关稀疏梯度的更多详细信息，请参阅“Notes”。

变量

**weight** ([*Tensor*](https://docs.pytorch.ac.cn/docs/stable/tensors.html#torch.Tensor "torch.Tensor")) – 模块的可学习权重，形状为 (num_embeddings, embedding_dim)，从 N(0,1) 初始化。

形状

- 输入: (∗)，包含要提取索引的任意形状的 IntTensor 或 LongTensor。

- 输出: (∗,H)，其中 * 是输入形状，H=embedding_dim。

注意

请注意，只有有限数量的优化器支持稀疏梯度：目前是 `optim.SGD` (CUDA 和 CPU)、`optim.SparseAdam` (CUDA 和 CPU) 以及 `optim.Adagrad` (CPU)

注意

当 `max_norm` 不是 `None` 时，[`Embedding`](https://docs.pytorch.ac.cn/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding "torch.nn.Embedding") 的 forward 方法将原地修改 `weight` 张量。由于需要进行梯度计算的张量不能被原地修改，因此在调用 [`Embedding`](https://docs.pytorch.ac.cn/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding "torch.nn.Embedding") 的 forward 方法之前，对 `Embedding.weight` 进行可微分操作需要克隆 `Embedding.weight`（当 `max_norm` 不是 `None` 时）。例如：

n, d, m = 3, 5, 7
embedding = nn.Embedding(n, d, max_norm=1.0)
W = torch.randn((m, d), requires_grad=True)
idx = torch.tensor([1, 2])
a = (
    embedding.weight.clone() @ W.t()
)  # weight must be cloned for this to be differentiable
b = embedding(idx) @ W.t()  # modifies weight in-place
out = a.unsqueeze(0) + b.unsqueeze(1)
loss = out.sigmoid().prod()
loss.backward()

示例

# an Embedding module containing 10 tensors of size 3

embedding = nn.Embedding(10, 3)

# a batch of 2 samples of 4 indices each

input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
embedding(input)
tensor([[[-0.0251, -1.6902,  0.7172],
         [-0.6431,  0.0748,  0.6969],
         [ 1.4970,  1.3448, -0.9685],
         [-0.3677, -2.7265, -0.1685]],

        [[ 1.4970,  1.3448, -0.9685],
         [ 0.4362, -0.4004,  0.9400],
         [-0.6431,  0.0748,  0.6969],
         [ 0.9124, -2.3616,  1.1151]]])

# example with padding_idx

embedding = nn.Embedding(10, 3, padding_idx=0)
input = torch.LongTensor([[0, 2, 0, 5]])
embedding(input)
tensor([[[ 0.0000,  0.0000,  0.0000],
         [ 0.1535, -2.0309,  0.9315],
         [ 0.0000,  0.0000,  0.0000],
         [-0.1655,  0.9897,  0.0635]]])

# example of changing `pad` vector

padding_idx = 0
embedding = nn.Embedding(3, 3, padding_idx=padding_idx)
embedding.weight
Parameter containing:
tensor([[ 0.0000,  0.0000,  0.0000],
        [-0.7895, -0.7089, -0.0364],
        [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
with torch.no_grad():
    embedding.weight[padding_idx] = torch.ones(3)
embedding.weight
Parameter containing:
tensor([[ 1.0000,  1.0000,  1.0000],
        [-0.7895, -0.7089, -0.0364],
        [ 0.6778,  0.5803,  0.2678]], requires_grad=True)

*classmethod* from_pretrained(*embeddings*, *freeze=True*, *padding_idx=None*, *max_norm=None*, *norm_type=2.0*, *scale_grad_by_freq=False*, *sparse=False*)[[source]](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/nn/modules/sparse.py#L216)

从给定的二维 FloatTensor 创建 Embedding 实例。

参数

- **embeddings** ([*Tensor*](https://docs.pytorch.ac.cn/docs/stable/tensors.html#torch.Tensor "torch.Tensor")) – 包含 Embedding 权重的 FloatTensor。第一维度传递给 Embedding 作为 `num_embeddings`，第二维度作为 `embedding_dim`。

- **freeze** ([*bool*](https://docs.pythonlang.cn/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) – 如果为 `True`，则张量在学习过程中不会被更新。等同于 `embedding.weight.requires_grad = False`。默认值：`True`

- **padding_idx** ([*int*](https://docs.pythonlang.cn/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) – 如果指定，`padding_idx`处的条目不计入梯度；因此，`padding_idx`处的嵌入向量在训练期间不会被更新，即它保持为一个固定的“pad”。

- **max_norm** ([*float*](https://docs.pythonlang.cn/3/library/functions.html#float "(in Python v3.14)")*,* *optional*) – 请参阅模块初始化文档。

- **norm_type** ([*float*](https://docs.pythonlang.cn/3/library/functions.html#float "(in Python v3.14)")*,* *optional*) – 请参见模块初始化文档。默认为 `2.0`。

- **scale_grad_by_freq** ([*bool*](https://docs.pythonlang.cn/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) – 请参见模块初始化文档。默认为 `False`。

- **sparse** ([*bool*](https://docs.pythonlang.cn/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) – 请参阅模块初始化文档。

示例

# FloatTensor containing pretrained weights

weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
embedding = nn.Embedding.from_pretrained(weight)

# Get embeddings for index 1

input = torch.LongTensor([1])
embedding(input)
tensor([[ 4.0000,  5.1000,  6.3000]])
