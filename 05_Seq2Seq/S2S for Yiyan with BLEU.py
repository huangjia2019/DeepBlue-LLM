# Step 1: 数据预处理
import jieba  # 使用jieba进行中文分词
import re  # 使用正则表达式对英文句子进行分词

# 读取句子
with open('05_Seq2Seq/all_sentences.txt', 'r', encoding='utf-8') as f:
    all_sentences = f.readlines()

# 创建用于存储中英文句子对的列表
sentences = []

# 对每个句子进行处理
for i in range(0, len(all_sentences), 2):
    # 对中文句子进行分词
    sentence_cn = ' '.join(jieba.cut(all_sentences[i].strip(), cut_all=False))
    # 对英文句子进行分词
    sentence_en = ' '.join(re.findall(r'\b\w+\b', all_sentences[i+1].strip()))
    # 构建句子对，分别添加<sos>和<eos>标记
    sentences.append([sentence_cn, '<sos> ' + sentence_en, sentence_en + ' <eos>'])

# 前5个句子对的显示
for s in sentences[:5]:
    print(s)

# Step 2: 构建词汇表
word_list_cn, word_list_en = [], []  # 初始化中英文单词列表
# 遍历每一个句子并将单词添加到单词列表中
for s in sentences:
    word_list_cn.extend(s[0].split())
    word_list_en.extend(s[1].split())
    word_list_en.extend(s[2].split())
# 去重得到不重复的单词列表
word_list_cn = list(set(word_list_cn))
word_list_en = list(set(word_list_en))

# Add special tokens to the vocabulary
word_list_cn = ['<pad>'] + word_list_cn
word_list_en = ['<pad>', '<sos>', '<eos>'] + word_list_en

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
print("中文词汇到索引的字典：", list(word2idx_cn.items())[:20]) # 中文词汇到索引
print("英文词汇到索引的字典：", list(word2idx_en.items())[:20]) # 英文词汇到索引

# Step 3: 构建数据集
import torch
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    def __init__(self, sentences, word2idx_cn, word2idx_en):
        self.sentences = sentences
        self.word2idx_cn = word2idx_cn
        self.word2idx_en = word2idx_en

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        # 将句子转换为索引
        sentence_cn = [self.word2idx_cn[word] for word in self.sentences[index][0].split()]
        sentence_en_in = [self.word2idx_en[word] for word in self.sentences[index][1].split()]
        sentence_en_out = [self.word2idx_en[word] for word in self.sentences[index][2].split()]
        return torch.tensor(sentence_cn), torch.tensor(sentence_en_in), torch.tensor(sentence_en_out)

# Collate function to pad sentences in a batch
def collate_fn(batch):
    # Sort the batch by the length of the sentences in descending order
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sentence_cn, sentence_en_in, sentence_en_out = zip(*batch)
    # Pad the sentences
    sentence_cn = nn.utils.rnn.pad_sequence(sentence_cn, padding_value=word2idx_cn['<pad>'])
    sentence_en_in = nn.utils.rnn.pad_sequence(sentence_en_in, padding_value=word2idx_en['<sos>'])
    sentence_en_out = nn.utils.rnn.pad_sequence(sentence_en_out, padding_value=word2idx_en['<eos>'])
    return sentence_cn, sentence_en_in, sentence_en_out

# 创建数据集
dataset = TranslationDataset(sentences, word2idx_cn, word2idx_en)
# 创建数据加载器，pass collate_fn to DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# Step 4: 构建模型
import torch.nn as nn
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
      
    def forward(self, input, hidden):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output.squeeze(0))
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size)
    
# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = Seq2Seq(voc_size_cn, 256, voc_size_en)
# Move the model to the device (GPU if available, otherwise CPU)
model = model.to(device)

# Step 5: 训练模型
from tqdm import tqdm
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

from torchtext.data.metrics import bleu_score
def compute_bleu(model, dataloader):
    model.eval()
    total_score = 0.0
    with torch.no_grad():
        for sentence_cn, sentence_en_in, sentence_en_out in dataloader:
            sentence_cn = sentence_cn.to(device)
            sentence_en_in = sentence_en_in.to(device)
            sentence_en_out = sentence_en_out.to(device)
            
            hidden = model.init_hidden(sentence_cn.size(1)).to(device)
            output, hidden = model(sentence_en_in, hidden)
            
            # Convert output to predicted tokens
            pred_tokens = output.argmax(2).detach().cpu().numpy().tolist()
            target_tokens = sentence_en_out.cpu().numpy().tolist()

            pred_sentences = [[str(token) for token in sentence] for sentence in pred_tokens]
            target_sentences = [[[str(token) for token in sentence]] for sentence in target_tokens]

            # Calculate BLEU score
            for pred_sentence, target_sentence in zip(pred_sentences, target_sentences):
                total_score += bleu_score([pred_sentence], [target_sentence])

    return total_score / len(dataloader)

for epoch in range(10):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Epoch {:03d}'.format(epoch + 1), leave=False, disable=False)
    for sentence_cn, sentence_en_in, sentence_en_out in progress_bar:
        sentence_cn = sentence_cn.to(device)
        sentence_en_in = sentence_en_in.to(device)
        sentence_en_out = sentence_en_out.to(device)
        
        hidden = model.init_hidden(sentence_cn.size(1)).to(device)
        optimizer.zero_grad()
        output, hidden = model(sentence_en_in, hidden)
        loss = criterion(output.view(-1, voc_size_en), sentence_en_out.view(-1))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(sentence_cn))})

    print(f"Epoch: {epoch+1}, Loss: {total_loss/len(dataloader)}")
    bleu = compute_bleu(model, dataloader)
    print(f"Bleu Score: {bleu}")


'''
你观察到的问题非常好！实际上，数据加载器（dataloader）的返回值确实是批次（batch）、序列长度（sequence length）、词汇表大小（vocabulary size）。在这个程序中，dataloader返回的是一批句子，其中包含了中文句子、英文输入句子和英文目标输出句子。其中，中文句子和英文句子都是已经转换为索引表示的。
在处理序列数据，如文本时，我们通常将一批数据整理为(batch_size, sequence_length)的形状，这是因为在一批数据中，每个数据项（这里是句子）的长度可能是不同的，所以我们需要使用填充（padding）来确保在同一批次中，所有数据项的长度都是相同的。
而模型的输出output的形状是(sequence_length, batch_size, vocabulary_size)。这是因为模型需要对输入句子的每个词输出一个词汇表大小的向量，其中每个元素代表该位置的词是输出词的概率。所以，output的第三维是词汇表大小。
因此，sentence_en_out的形状为(sequence_length, batch_size)，而output的形状为(sequence_length, batch_size, vocabulary_size)是符合预期的。output的每个词都是一个大小为词汇表大小的概率分布，而sentence_en_out的每个词都是一个具体的词（即一个具体的索引）。所以它们的形状是不同的。
当你在计算损失函数时，你需要调整output和sentence_en_out的形状使其匹配。在这个例子中，使用了如下代码来调整形状：
loss = criterion(output.view(-1, voc_size_en), sentence_en_out.view(-1))
这里的.view(-1, voc_size_en)将output的形状从(sequence_length, batch_size, vocabulary_size)改为了(sequence_length * batch_size, vocabulary_size)，.view(-1)将sentence_en_out的形状从(sequence_length, batch_size)改为了(sequence_length * batch_size,)。这样两者的形状就匹配了，可以用于计算损失函数。

是的，你的理解是正确的。这个变化的确是由于 RNN 的设计引起的。
通常情况下，我们在准备批量数据时，会按照 (batch_size, sequence_length) 的形状来组织，其中每个元素是单词的索引。这种方式使我们能够一次处理多个句子，并通过填充操作保证每个句子的长度一致。
然而，当这些数据被传递到 RNN（或 LSTM、GRU 等其他递归神经网络）时，RNN 的输入需要是 (sequence_length, batch_size, input_size) 的形状。这是因为 RNN 是通过时间步（即序列的长度）来执行操作的，需要在每个时间步获取批量中所有句子的当前单词。
所以，在准备数据和模型训练这两个步骤中，我们需要进行这种形状的转换。在 PyTorch 中，我们通常使用 nn.utils.rnn.pack_padded_sequence 和 nn.utils.rnn.pad_packed_sequence 等函数来在两种形状间转换，同时处理变长序列的情况。
在模型的输出中，形状 (sequence_length, batch_size, vocabulary_size) 是因为模型对输入序列中的每个单词都会输出一个大小为词汇表大小的向量，代表每个词可能是该位置输出的概率。
总的来说，这种形状的变化是由于我们在数据处理和模型设计中对性能和效率的考虑。

这种转换对于损失(loss)计算以及BLEU得分的计算基本没有影响。
损失计算：在模型的输出和目标输出进行损失计算时，我们通常会将它们展平到一维，然后计算损失。例如，使用交叉熵损失(CrossEntropyLoss)时，我们会将模型输出的形状从 (sequence_length, batch_size, vocabulary_size) 转换为 (sequence_length * batch_size, vocabulary_size)，将目标输出的形状从 (sequence_length, batch_size) 转换为 (sequence_length * batch_size,)。因此，数据的形状并不会影响损失的计算。
BLEU得分计算：BLEU得分是根据模型生成的翻译和真实的目标翻译进行计算的。在计算BLEU得分时，我们需要将模型的输出从索引转换回单词，然后与目标输出（也是单词）进行比较。因此，只要我们能正确地将索引转换为单词，数据的形状也不会影响BLEU得分的计算。
需要注意的是，在进行这些操作时，我们需要确保各种操作是在正确的维度上进行的，且数据在各步骤间能够正确对应。例如，在计算损失时，模型的每个输出必须与目标输出的正确单词对应；在计算BLEU得分时，生成的翻译和目标翻译的单词顺序必须一致。

'''