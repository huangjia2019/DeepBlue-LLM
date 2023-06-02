#----------------------------------------------------
# 生成式预训练语言模型：理论与实战
# 深蓝学院 课程 
# 课程链接：https://www.shenlanxueyuan.com/course/620
#
# 作者 **黄佳**
#----------------------------------------------------
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import torch.nn as nn
from Transformer_Model import Transformer

from CorpusLoader import CorpusLoader
from torch.utils.data import DataLoader

corpus_loader = CorpusLoader('05_Seq2Seq/all_sentences.txt')
corpus_loader.process_sentences()
corpus_loader.build_vocab()

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sentence_cn, sentence_en_in, sentence_en_out = zip(*batch)
    sentence_cn = nn.utils.rnn.pad_sequence(sentence_cn, padding_value=corpus_loader.word2idx_cn['<pad>'],batch_first=True)
    sentence_en_in = nn.utils.rnn.pad_sequence(sentence_en_in, padding_value=corpus_loader.word2idx_en['<pad>'],batch_first=True)
    sentence_en_out = nn.utils.rnn.pad_sequence(sentence_en_out, padding_value=corpus_loader.word2idx_en['<pad>'],batch_first=True)
    return sentence_cn, sentence_en_in, sentence_en_out


dataset = corpus_loader.create_dataset()
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

import torch # 导入torch
import torch.optim as optim # 导入优化器
device = "cuda" if torch.cuda.is_available() else "cpu"

# 创建模型实例并选择优化器
model = Transformer(corpus_loader).to(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=corpus_loader.word2idx_en['<pad>']) # 忽略padding的损失
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 进行训练
for epoch in range(100):
    for i, (enc_inputs, dec_inputs, target_batch) in enumerate(dataloader):
        enc_inputs, dec_inputs, target_batch = enc_inputs.to(device), dec_inputs.to(device), target_batch.to(device)
        optimizer.zero_grad()
        outputs, _, _, _ = model(enc_inputs, dec_inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), target_batch.view(-1))
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"Epoch[{epoch+1}/100], Step[{i+1}/{len(dataloader)}], Loss: {loss.item()}")

    if (epoch + 1) % 20 == 0: # 打印损失
        print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")


# # 创建一个大小为1的批次，目标语言序列dec_inputs在测试阶段，仅包含句子开始符号<sos>
# enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size=1,test_batch=True) 
# enc_inputs, dec_inputs, target_batch = enc_inputs.to(device), dec_inputs.to(device), target_batch.to(device)
# print("编码器输入:", enc_inputs) # 打印编码器输入
# print("解码器输入:", dec_inputs) # 打印解码器输入
# print("目标数据:", target_batch) # 打印目标数据
# predict, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs) # 用模型进行翻译
# predict = predict.view(-1, len(corpus.tgt_vocab)) # 将预测结果维度重塑
# predict = predict.data.max(1, keepdim=True)[1] # 找到每个位置概率最大的词汇的索引
# # 解码预测的输出，将所预测的目标句子中的索引转换为单词
# translated_sentence = [corpus.tgt_idx2word[idx.item()] for idx in predict.squeeze()]
# # 将输入的源语言句子中的索引转换为单词
# input_sentence = ' '.join([corpus.src_idx2word[idx.item()] for idx in enc_inputs[0]])
# print(input_sentence, '->', translated_sentence) # 打印原始句子和翻译后的句子

