# 构建训练句子集，每一个句子包含中文，英文（解码器输入）和翻译成英文后目标输出三个部分
sentences = [
    ['咖哥 喜欢 小冰', '<sos> KaGe likes XiaoBing', 'KaGe likes XiaoBing <eos>'],
    ['我 爱 学习 人工智能', '<sos> I love studying AI', 'I love studying AI <eos>'],
    ['深度 学习 改变 世界', '<sos> Deep learning YYDS', 'Deep learning YYDS <eos>'],
    ['自然 语言 处理 很 强大', '<sos> NLP is so powerful', 'NLP is so powerful <eos>'],
    ['神经网络 非常 复杂', '<sos> Neural-Net are complex', 'Neural-Net are complex <eos>']]

word_list_cn, word_list_en = [], []  # 初始化中英文单词列表
# 遍历每一个句子并将单词添加到单词列表中
for s in sentences:
    word_list_cn.extend(s[0].split())
    word_list_en.extend(s[1].split())
    word_list_en.extend(s[2].split())

# 去重得到不重复的单词列表
word_list_cn = list(set(word_list_cn))
word_list_en = list(set(word_list_en))

# 构建单词到索引的映射
word2idx_cn = {w: i for i, w in enumerate(word_list_cn)}
word2idx_en = {w: i for i, w in enumerate(word_list_en)}

# 构建索引到单词的映射
idx2word_cn = {i: w for i, w in enumerate(word_list_cn)}
idx2word_en = {i: w for i, w in enumerate(word_list_en)}

# 计算词汇表的大小
voc_size_cn = len(word_list_cn)
voc_size_en = len(word_list_en)

print("句子数量：", len(sentences)) # 打印句子数
print("中文词汇表大小：", voc_size_cn) #打印中文词汇表大小
print("英文词汇表大小：", voc_size_en) #打印英文词汇表大小
print("中文词汇到索引的字典：", word2idx_cn) # 中文词汇到索引
print("英文词汇到索引的字典：", word2idx_en) # 英文词汇到索引


import numpy as np # 导入numpy
import torch # 导入torch
import random # 导入random库
# 定义一个函数，随机选择一个句子和词典生成输入、输出和目标数据
def make_data(sentences):
    # 随机选择一个句子进行训练
    random_sentence = random.choice(sentences)    
    # 将输入句子中的单词转换为对应的索引
    encoder_input = np.array([[word2idx_cn[n] for n in random_sentence[0].split()]])
    # 将输出句子中的单词转换为对应的索引
    decoder_input = np.array([[word2idx_en[n] for n in random_sentence[1].split()]])
    # 将目标句子中的单词转换为对应的索引
    target = np.array([[word2idx_en[n] for n in random_sentence[2].split()]])
    # 将输入、输出和目标批次转换为LongTensor
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input)
    target = torch.LongTensor(target)
    return encoder_input, decoder_input, target

# 使用make_data函数生成输入、输出和目标张量
encoder_input, decoder_input, target = make_data(sentences)
for s in sentences: # 获取原始句子
    if all([word2idx_cn[w] in encoder_input[0] for w in s[0].split()]):
        original_sentence = s
        break
print("原始句子:", original_sentence) # 打印原始句子
print("编码器输入张量的形状:", encoder_input.shape)  # 打印输入张量形状
print("解码器输入张量的形状:", decoder_input.shape) # 打印输出张量形状
print("目标张量的形状:", target.shape) # 打印目标张量形状
print("编码器输入张量:", encoder_input) # 打印输入张量
print("解码器输入张量:", decoder_input) # 打印输出张量
print("目标张量:", target) # 打印目标张量

    
import torch.nn as nn # 导入torch.nn库
# 创建一个Attention类，用于计算注意力权重
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, decoder_context, encoder_context):
        # 计算decoder_context和encoder_context的点积，得到注意力分数
        scores = torch.matmul(decoder_context, encoder_context.transpose(-2, -1))
        # 归一化分数
        attn_weights = nn.functional.softmax(scores, dim=-1)
        # 将注意力权重乘以encoder_context，得到加权的上下文向量
        context = torch.matmul(attn_weights, encoder_context)
        return context, attn_weights
    

# 定义编码器类，继承自nn.Module
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()       
        self.hidden_size = hidden_size # 设置隐藏层大小       
        self.embedding = nn.Embedding(input_size, hidden_size) # 创建词嵌入层       
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True) # 创建RNN层
    # 定义前向传播函数
    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs) # 将输入转换为嵌入向量       
        output, hidden = self.rnn(embedded, hidden) # 将嵌入向量输入RNN层并获取输出
        return output, hidden

# 定义解码器类，继承自nn.Module
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()       
        self.hidden_size = hidden_size # 设置隐藏层大小       
        self.embedding = nn.Embedding(output_size, hidden_size) # 创建词嵌入层
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)  # 创建RNN层       
        self.out = nn.Linear(hidden_size, output_size) # 创建线性输出层
    # 定义前向传播函数
    def forward(self, inputs, hidden):       
        embedded = self.embedding(inputs) # 将输入转换为嵌入向量       
        output, hidden = self.rnn(embedded, hidden) # 将嵌入向量输入RNN层并获取输出       
        output = self.out(output) # 使用线性层生成最终输出
        return output, hidden

class DecoderWithAttention(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size # 设置隐藏层大小
        self.embedding = nn.Embedding(output_size, hidden_size) # 创建词嵌入层
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True) # 创建RNN层
        self.attention = Attention()  # 创建注意力层
        self.out = nn.Linear(2 * hidden_size, output_size)  # 修改线性输出层，考虑隐藏状态和上下文向量

    def forward(self, inputs, hidden, encoder_outputs):
        embedded = self.embedding(inputs)  # 将输入转换为嵌入向量
        rnn_output, hidden = self.rnn(embedded, hidden)  # 将嵌入向量输入RNN层并获取输出 
        context, attn_weights = self.attention(rnn_output, encoder_outputs)  # 计算注意力上下文向量
        output = torch.cat((rnn_output, context), dim=-1)  # 将上下文向量与解码器的输出拼接
        output = self.out(output)  # 使用线性层生成最终输出
        return output, hidden, attn_weights
    
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 设置设备，检查是否有GPU
n_hidden = 128 # 设置隐藏层数量

# 创建编码器和解码器
encoder = Encoder(voc_size_cn, n_hidden)
decoder = DecoderWithAttention(n_hidden, voc_size_en)
print('编码器结构：', encoder)  # 打印编码器的结构
print('解码器结构：', decoder)  # 打印解码器的结构


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        # 初始化编码器和解码器
        self.encoder = encoder
        self.decoder = decoder
    # 定义前向传播函数
    def forward(self, source, hidden, target):
        # 将输入序列通过编码器并获取输出和隐藏状态
        encoder_output, encoder_hidden = self.encoder(source, hidden)
        # 将编码器的隐藏状态传递给解码器作为初始隐藏状态
        decoder_hidden = encoder_hidden
        # 将目标序列通过解码器并获取输出 -  此处更新解码器调用
        decoder_output, _, attn_weights = self.decoder(target, decoder_hidden, encoder_output) 
        return decoder_output, attn_weights

# 创建Seq2Seq模型
model = Seq2Seq(encoder, decoder)
print('S2S模型结构:', model)  # 打印模型的结构


# 定义训练函数
def train_seq2seq(model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        encoder_input, decoder_input, target = make_data(sentences) # 训练数据的创建
        hidden = torch.zeros(1, encoder_input.size(0), n_hidden) # 初始化隐藏状态      
        optimizer.zero_grad()# 梯度清零        
        output, _ = model(encoder_input, hidden, decoder_input) # 获取模型输出        
        loss = criterion(output.view(-1, voc_size_en), target.view(-1)) # 计算损失        
        if (epoch + 1) % 40 == 0: # 打印损失
            print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")         
        loss.backward()# 反向传播        
        optimizer.step()# 更新参数

# 训练模型
epochs = 800 # 训练轮次
criterion = nn.CrossEntropyLoss() # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 优化器
train_seq2seq(model, criterion, optimizer, epochs) # 调用函数训练模型


import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["font.family"]=['SimHei'] # 用来设定字体样式
plt.rcParams['font.sans-serif']=['SimHei'] # 用来设定无衬线字体样式
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

def visualize_attention(source_sentence, predicted_sentence, attn_weights):
    """
    可视化注意力权重矩阵。
    :param source_sentence: 源句子，字符串格式。
    :param predicted_sentence: 预测句子，字符串列表格式。
    :param attn_weights: 注意力权重矩阵，形状为 (len(predicted_sentence), len(source_sentence))
    """
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(attn_weights, annot=True, cbar=False, xticklabels=source_sentence.split(), yticklabels=predicted_sentence, cmap="Greens")

    # 设置边框
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(2)

    plt.xlabel("源序列")
    plt.ylabel("目标序列")
    # plt.show()


    # 定义测试函数
def test_seq2seq(model, source_sentence, word_dict, number_dict):
    # 将输入句子转换为索引
    encoder_input = np.array([[word2idx_cn[n] for n in source_sentence.split()]])
    # 构建输出句子的索引，以'<sos>'开始，后面跟'<eos>'，长度与输入句子相同
    decoder_input = np.array([word2idx_en['<sos>']] + [word_dict['<eos>']]*(len(encoder_input[0])-1))
    # 转换为LongTensor
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input).unsqueeze(0) # 增加一维    
    hidden = torch.zeros(1, encoder_input.size(0), n_hidden) # 初始化隐藏状态    
    predict, attn_weights= model(encoder_input, hidden, decoder_input) # 获取模型输出    
    predict = predict.data.max(2, keepdim=True)[1] # 获取最大概率的索引
    # 打印输入句子和预测的句子
    print(source_sentence, '->', [number_dict[n.item()] for n in predict.squeeze()])
    
    # 可视化注意力权重
    attn_weights = attn_weights.squeeze(0).cpu().detach().numpy()
    visualize_attention(source_sentence, [number_dict[n.item()] for n in predict.squeeze()], attn_weights)

# 测试模型
test_seq2seq(model, '咖哥 喜欢 小冰', word2idx_en, idx2word_en)  
test_seq2seq(model, '自然 语言 处理 很 强大', word2idx_en, idx2word_en)  