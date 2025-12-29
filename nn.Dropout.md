# nn.Dropout

*class* torch.nn.Dropout(*p=0.5*, *inplace=False*)[[source]](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/modules/dropout.py#L35)

During training, randomly zeroes some of the elements of the input tensor with probability `p`.

The zeroed elements are chosen independently for each forward call and are sampled from a Bernoulli distribution.

Each channel will be zeroed out independently on every forward call.

This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons as described in the paper [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580) .

Furthermore, the outputs are scaled by a factor of 1−p1​ during training. This means that during evaluation the module simply computes an identity function.

Parameters

- **p** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) – probability of an element to be zeroed. Default: 0.5

- **inplace** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If set to `True`, will do this operation in-place. Default: `False`

Shape:

- Input: (∗). Input can be of any shape

- Output: (∗). Output is of the same shape as input

Examples:

m = nn.Dropout(p=0.2)
input = torch.randn(20, 16)
output = m(input)

输出的结果：

![](https://picx.zhimg.com/v2-eac99ebfbe58dc3a8fc34625b78590b7_1440w.jpg)

我们会发现：

1. 有一部分的值变为了0，这些值大约占据总数的0.2。
2. 其它非0参数都**除以0.8**，使得值变大了。比如：`0.3514 / 0.8 = 0.4392`，`-1.0317 / 0.8 = -1.2896`。

## **Dropout的位置**

一般来说，我们在实现的神级网络中这么定义：

```text
 self.dropout = nn.Dropout(0.3)
```

但是具体在哪里使用是个问题。

一般来说，Dropout使用位置是在隐藏层之间的节点上，具体来说，就是在[全连接层](https://zhida.zhihu.com/search?content_id=232938406&content_type=Article&match_order=1&q=%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82&zhida_source=entity)之间放置Dropout来避免过拟合：

```text
 import torch
 import torch.nn as nn
 ​
 class Net(nn.Module):
     def __init__(self):
         super(Net, self).__init__()
         self.fc1 = nn.Linear(in_features, hidden_size)
         self.dropout = nn.Dropout(dropout_prob)
         self.fc2 = nn.Linear(hidden_size, out_features)

     def forward(self, x):
         x = self.fc1(x)
         x = self.dropout(x)
         x = torch.relu(x)
         x = self.fc2(x)
         return x
```

比如上面得这个例子，dropout被放置在fc1和fc2之间。
