import torch # 导入torch
import torch.nn.functional as F # 导入nn.functional

# 创建一个形状为 (batch_size, seq_len, feature_dim) 的张量 x
x = torch.randn(2, 3, 4) # (batch_size=2, seq_len=3, feature_dim=4)

# 定义头数和每个头的维度
num_heads = 2
head_dim = 2
# feature_dim 必须是 num_heads * head_dim 的整数倍
assert x.size(-1) == num_heads * head_dim

# 定义线性层用于将 x 转换为 Q, K, V 向量
linear_q = torch.nn.Linear(4, 4)
linear_k = torch.nn.Linear(4, 4)
linear_v = torch.nn.Linear(4, 4)

# 通过线性层计算 Q, K, V
Q = linear_q(x) # (batch_size=2, seq_len=3, Q_dim=4)
K = linear_k(x) # (batch_size=2, seq_len=3, K_dim=4)
V = linear_v(x) # (batch_size=2, seq_len=3, V_dim=4)

# 将 Q, K, V 分割成 num_heads 个头
def split_heads(tensor, num_heads):
    batch_size, seq_len, feature_dim = tensor.size()
    head_dim = feature_dim // num_heads
    return tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
Q = split_heads(Q, num_heads) # (batch_size=2, num_heads=2, seq_len=3, head_dim=2)
K = split_heads(K, num_heads) # (batch_size=2, num_heads=2, seq_len=3, head_dim=2)
V = split_heads(V, num_heads) # (batch_size=2, num_heads=2, seq_len=3, head_dim=2)


# 计算 Q 和 K 的点积，作为相似度分数,也就是自注意力原始权重
raw_weights = torch.matmul(Q, K.transpose(-2, -1)) # (batch_size=2, num_heads=2, seq_len=3, seq_len=3)

# 自注意力原始权重进行缩放
scale_factor = K.size(-1) ** 0.5
scaled_weights = raw_weights / scale_factor # (batch_size=2, num_heads=2, seq_len=3, seq_len=3)

# 对缩放后的权重进行 softmax 归一化，得到注意力权重
attn_weights = F.softmax(scaled_weights, dim=-1)  # (batch_size=2, num_heads=2, seq_len=3, seq_len=3)

# 将注意力权重应用于 V 向量，计算加权和，得到加权信息
attn_outputs = torch.matmul(attn_weights, V) # (batch_size=2, num_heads=2, seq_len=3, head_dim=2)

# 将所有头的结果拼接起来
def combine_heads(tensor, num_heads):
    batch_size, num_heads, seq_len, head_dim = tensor.size()
    feature_dim = num_heads * head_dim
    return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, feature_dim)
attn_outputs = combine_heads(attn_outputs, num_heads)  # (batch_size=2, seq_len=3, feature_dim=4)

# 对拼接后的结果进行线性变换
linear_out = torch.nn.Linear(4, 4)
attn_outputs = linear_out(attn_outputs) # (batch_size=2, seq_len=3, output_dim=4)
print("加权信息:", attn_outputs)