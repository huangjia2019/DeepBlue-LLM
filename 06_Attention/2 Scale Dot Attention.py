import torch # 导入torch
import torch.nn.functional as F # 导入nn.functional

# 1. 创建两个张量 x1 和 x2
x1 = torch.randn(2, 3, 4) # 形状(batch_size, seq_len1, feature_dim)
x2 = torch.randn(2, 4, 4) # 形状(batch_size, seq_len2, feature_dim)

# 2. 计算点积，结果形状为 (batch_size, seq_len1, seq_len2)
raw_weights  = torch.matmul(x1, x2.transpose(1, 2))

# 3. 将原始权重进行缩放（可选）
scaling_factor = x1.size(-1) ** 0.5
scaled_weights = raw_weights  / scaling_factor

# 4. 应用Softmax函数，使结果的值在 0 和 1 之间，且每一行的和为 1
attention_weights = F.softmax(scaled_weights, dim=-1)

# 5. 与 x2 相乘，得到注意力分布的加权和, 形状为(batch_size, seq_len1, feature_dim)
attention_outputs = torch.matmul(attention_weights, x2) 