import torch
import torch.nn.functional as F

# 1. 创建 Query、Key 和 Value 张量
q = torch.randn(2, 3, 4) # 形状(batch_size, seq_len1, feature_dim)
k = torch.randn(2, 4, 4) # 形状(batch_size, seq_len2, feature_dim)
v = torch.randn(2, 4, 4) # 形状(batch_size, seq_len2, feature_dim)

# 2. 计算点积，得到原始权重，形状为(batch_size, seq_len1, seq_len2)
raw_weights = torch.matmul(q, k.transpose(1, 2))
print("原始权重:", raw_weights)

# 3. 将原始权重进行缩放（可选）
scaling_factor = q.size(-1) ** 0.5
scaled_weights = raw_weights / scaling_factor
print("缩放后权重:", scaled_weights)

# 4. 应用Softmax函数，使结果的值在 0 和 1 之间，且每一行的和为 1
attention_weights = F.softmax(scaled_weights, dim=-1)
print("注意力权重:", attention_weights)

# 5. 与Value相乘，得到注意力分布的加权和, 形状为(batch_size, seq_len1, feature_dim)
attention_outputs = torch.matmul(attention_weights, v)
print("注意力输出:", attention_outputs)