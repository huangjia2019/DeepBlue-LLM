import torch # 导入torch
import torch.nn.functional as F # 导入nn.functional
# 1. 创建两个张量 x1 和 x2
x1 = torch.randn(2, 3, 4) # 形状(batch_size, seq_len1, feature_dim)
x2 = torch.randn(2, 5, 4) # 形状(batch_size, seq_len2, feature_dim)
# 2. 计算原始权重
raw_weights = torch.bmm(x1, x2.transpose(1, 2))
# - torch.bmm 是批次矩阵乘法，它将矩阵乘法应用于批次中的矩阵。
# 3. 对原始权重进行 softmax 归一化
attention_weights = F.softmax(raw_weights, dim=2)
# 4. 计算加权和
attention_output = torch.bmm(attention_weights, x2)

# 创建两个张量 x1 x2
import torch # 导入torch
x1 = torch.randn(2, 3, 4) # 形状(batch_size, seq_len1, feature_dim)
x2 = torch.randn(2, 5, 4) # 形状(batch_size, seq_len2, feature_dim)
print("x1:", x1)
print("x2:", x2)

# 计算点积，得到原始权重
raw_weights = torch.bmm(x1, x2.transpose(1, 2))
# 结果形状为 (batch_size, seq_len1, seq_len2)
print("原始权重:", raw_weights)

# 应用Softmax函数，使结果的值在 0 和 1 之间，且每一行的和为 1
import torch.nn.functional as F
# 设置打印选项，禁用科学计数法
torch.set_printoptions(sci_mode=False)
attention_weights = F.softmax(raw_weights, dim=-1)
print("注意力权重:", attention_weights)

# 与 x2 相乘，得到注意力分布的加权和，形状为 (batch_size, seq_len1, feature_dim)
attention_output = torch.bmm(attention_weights, x2)
print("注意力输出:", attention_output)