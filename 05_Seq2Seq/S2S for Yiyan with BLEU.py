# Step 1: 数据预处理
import jieba  # 使用jieba进行中文分词
import re  # 使用正则表达式对英文句子进行分词

# 读取句子
with open('01_Data/yiyan_translation/all_sentences.txt', 'r', encoding='utf-8') as f:
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
class Seq2Seq(nn.Module): # 请同学们按照课程中介绍的S2S-encoder和decoder架构重构下面的模型
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