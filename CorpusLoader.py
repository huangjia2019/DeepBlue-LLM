#----------------------------------------------------
# 生成式预训练语言模型：理论与实战
# 深蓝学院 课程 
# 课程链接：https://www.shenlanxueyuan.com/course/620
#
# 作者 **黄佳**
#----------------------------------------------------
import torch
from collections import Counter



# WikiCorpus语料库类
class WikiCorpus:
    def __init__(self, sentences, max_seq_len=256):
        self.sentences = sentences
        self.seq_len = max_seq_len
        self.vocab = self.create_vocabularies()
        self.vocab_size = len(self.vocab)
        self.idx2word = {v: k for k, v in self.vocab.items()}

    def create_vocabularies(self):
        # counter = Counter(word for sentence in self.sentences for word in sentence.split())
        # vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, **{word: i+3 for i, word in enumerate(counter)}}
        with open("shared_vocab.txt", "r") as f:
            vocab = {line.split()[0]: int(line.split()[1]) for line in f}
        return vocab   


    def make_batch(self, batch_size):
        input_batch, target_batch = [], []

        # 随机选择句子索引
        sentence_indices = torch.randperm(len(self.sentences))[:batch_size]
        for index in sentence_indices:
            sentence = self.sentences[index]
            words = sentence.split()[:self.seq_len - 2]  # 截断句子,确保长度不超过max_seq_len - 2（为了留出<sos>和<eos>）
            seq = [self.vocab['<sos>']] + [self.vocab[word] for word in words] + [self.vocab['<eos>']]

            # 对序列进行填充
            seq += [self.vocab['<pad>']] * (self.seq_len - len(seq))

            # 将处理好的序列添加到批次中
            input_batch.append(seq[:-1])
            target_batch.append(seq[1:])

        # 将批次转换为LongTensor类型
        input_batch = torch.LongTensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        return input_batch, target_batch


class MovieCorpus:
    def __init__(self, sentences, max_seq_len=256):
        self.sentences = sentences
        self.seq_len = max_seq_len
        self.vocab = self.create_vocabularies()
        self.vocab_size = len(self.vocab)
        self.idx2word = {v: k for k, v in self.vocab.items()}

    def create_vocabularies(self):
        with open("shared_vocab.txt", "r") as f:
            vocab = {line.split()[0]: int(line.split()[1]) for line in f}
        return vocab   

    def make_batch(self, batch_size):
        input_batch, target_batch = [], []

        # 随机选择句子索引
        sentence_indices = torch.randperm(len(self.sentences) - 2)[:batch_size]  # -2 以确保不超过句子列表的倒数第二个元素
        for index in sentence_indices:
            sentence = self.sentences[index] + " " + self.sentences[index + 1]  # 合并后面两个句子
            words = sentence.split()[:self.seq_len - 2]  # 截断句子,确保长度不超过max_seq_len - 2（为了留出<sos>和<eos>）

            seq = [self.vocab['<sos>']] + [self.vocab[word] for word in words] + [self.vocab['<eos>']]

            # 对序列进行填充
            seq += [self.vocab['<pad>']] * (self.seq_len - len(seq))

            # 将处理好的序列添加到批次中
            input_batch.append(seq[:-1])
            target_batch.append(seq[1:])

        # 将批次转换为LongTensor类型
        input_batch = torch.LongTensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        return input_batch, target_batch


class DialogCorpus:
    def __init__(self, sentences, max_seq_len=256):
        self.sentences = sentences
        self.seq_len = max_seq_len
        self.vocab = self.create_vocabularies()
        self.vocab_size = len(self.vocab)
        self.idx2word = {v: k for k, v in self.vocab.items()}

    def create_vocabularies(self):
        # counter = Counter(word for sentence in self.sentences for word in sentence.split())
        # vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, **{word: i+3 for i, word in enumerate(counter)}}
        with open("shared_vocab.txt", "r") as f:
            vocab = {line.split()[0]: int(line.split()[1]) for line in f}        
        return vocab

    def make_batch(self, batch_size):
        input_batch, target_batch = [], []

        # 随机选择句子索引
        sentence_indices = torch.randperm(len(self.sentences) - 1)[:batch_size]  # -1 以确保不超过句子列表的最后一个元素
        for index in sentence_indices:
            input_sentence = self.sentences[index]
            target_sentence = self.sentences[index + 1]

            input_seq = [self.vocab['<sos>']] + [self.vocab[word] for word in input_sentence.split()] + [self.vocab['<eos>']]
            target_seq = [self.vocab['<sos>']] + [self.vocab[word] for word in target_sentence.split()] + [self.vocab['<eos>']]

            # 对序列进行填充
            input_seq += [self.vocab['<pad>']] * (self.seq_len - len(input_seq))
            target_seq += [self.vocab['<pad>']] * (self.seq_len - len(target_seq))

            input_batch.append(input_seq[:self.seq_len])
            target_batch.append(target_seq[:self.seq_len])

        # 将批次转换为LongTensor类型
        input_batch = torch.LongTensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        return input_batch, target_batch


import jieba
import re
class CorpusLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.sentences = []
        self.word_list_cn = ['<pad>']
        self.word_list_en = ['<pad>', '<sos>', '<eos>']

    def process_sentences(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            all_sentences = f.readlines()

        for i in range(0, len(all_sentences), 2):
            sentence_cn = ' '.join(jieba.cut(all_sentences[i].strip(), cut_all=False))
            sentence_en = ' '.join(re.findall(r'\b\w+\b', all_sentences[i+1].strip()))
            self.sentences.append([sentence_cn, '<sos> ' + sentence_en + ' <eos>'])

    def build_vocab(self):
        for s in self.sentences:
            self.word_list_cn.extend(s[0].split())
            self.word_list_en.extend(s[1].split())

        self.word_list_cn = list(set(self.word_list_cn))
        self.word_list_en = list(set(self.word_list_en))

        # self.word_list_cn = sorted(list(set(self.word_list_cn)), key=lambda x: self.word_list_cn.index(x))
        # self.word_list_en = sorted(list(set(self.word_list_en)), key=lambda x: self.word_list_en.index(x))

        # Ensure special tokens have the desired indices
        for token in ['<pad>', '<sos>', '<eos>']:
            if token in self.word_list_en:
                self.word_list_en.remove(token)
            self.word_list_en.insert(0, token)
        
        self.word2idx_cn = {w: i for i, w in enumerate(self.word_list_cn)}
        self.word2idx_en = {w: i for i, w in enumerate(self.word_list_en)}
        
        self.idx2word_cn = {i: w for i, w in enumerate(self.word_list_cn)}
        self.idx2word_en = {i: w for i, w in enumerate(self.word_list_en)}

        self.src_vocab = len(self.word2idx_cn)
        self.tgt_vocab = len(self.word2idx_en)

        self.src_len = max(len(sentence.split()) for sentence, _ in self.sentences)
        self.tgt_len = max(len(sentence.split()) for _, sentence in self.sentences)

    def create_dataset(self):
        return TranslationDataset(self.sentences, self.word2idx_cn, self.word2idx_en)


import torch
from torch.utils.data import Dataset
class TranslationDataset(Dataset):
    def __init__(self, sentences, word2idx_cn, word2idx_en):
        self.sentences = sentences
        self.word2idx_cn = word2idx_cn
        self.word2idx_en = word2idx_en

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence_cn = [self.word2idx_cn[word] for word in self.sentences[index][0].split()]
        sentence_en = [self.word2idx_en[word] for word in self.sentences[index][1].split()]
        sentence_en_in = sentence_en[:-1]  # remove <eos>
        sentence_en_out = sentence_en[1:]  # remove <sos>
        return torch.tensor(sentence_cn), torch.tensor(sentence_en_in), torch.tensor(sentence_en_out)

