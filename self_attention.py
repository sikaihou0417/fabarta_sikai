import torch
import numpy as np
#输入X，维度为(3,4)
dim=4
X=np.random.randn(3,dim)
#可学习矩阵W_q,W_k,W_v，维度为(dim,dim)
W_q=np.random.randn(dim,dim)
W_k=np.random.randn(dim,dim)
W_v=np.random.randn(dim,dim)
#计算QKV
Q=X@W_q
K=X@W_k
V=X@W_v
#注意力权重
attention_weight=Q@K.T/np.sqrt(dim)
#注意力权重归一化
attention_weight=torch.softmax(np.array(attention_weight),axis=1)
#输出
output=attention_weight@V
print(output)
