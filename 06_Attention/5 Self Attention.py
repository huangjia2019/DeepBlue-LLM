import torch # 导入torch
import torch.nn.functional as F # 导入nn.functional

# 创建一个形状为(batch_size, seq_len, feature_dim) 的张量 x
x = torch.randn(2, 3, 4) # (batch_size=2, seq_len=3, feature_dim=4)
# 计算自注意力原始权重
raw_weights = torch.bmm(x, x.transpose(1, 2)) # (batch_size=2, seq_len=3, seq_len=3)
# 对原始权重进行 softmax 归一化
attn_weights = F.softmax(raw_weights, dim=2) # (batch_size=2, seq_len=3, seq_len=3)
# 计算自注意力加权和
attn_outputs = torch.bmm(attn_weights, x)  # (batch_size=2, seq_len=3, feature_dim=4)


# 创建一个形状为 (batch_size, seq_len, feature_dim) 的张量 x
x = torch.randn(2, 3, 4) # (batch_size=2, seq_len=3, feature_dim=4)
# 定义三个线性层用于将 x 转换为 Q, K, V 向量
linear_q = torch.nn.Linear(4, 4) 
linear_k = torch.nn.Linear(4, 4)
linear_v = torch.nn.Linear(4, 4)
# 计算 Q, K, V
Q = linear_q(x) # (batch_size=2, seq_len=3, Q_dim=4)
K = linear_k(x) # (batch_size=2, seq_len=3, K_dim=4)
V = linear_v(x) # (batch_size=2, seq_len=3, V_dim=4)
# 计算 Q 和 K 的点积，作为相似度分数,也就是自注意力原始权重
raw_weights = torch.bmm(Q, K.transpose(1, 2)) # (batch_size=2, seq_len=3, seq_len=3)
# 将自注意力原始权重进行缩放
scale_factor = K.size(-1) ** 0.5  # 这里是4 ** 0.5
scaled_weights = raw_weights / scale_factor # (batch_size=2, seq_len=3, seq_len=3)
# 对缩放后的权重进行 softmax 归一化，得到注意力权重
attn_weights = F.softmax(scaled_weights, dim=2)
# 将注意力权重应用于 V 向量，计算加权和，得到加权信息
attn_outputs = torch.bmm(attn_weights, V) # (batch_size=2, seq_len=3, V_dim=4)
print("加权信息:", attn_outputs) 