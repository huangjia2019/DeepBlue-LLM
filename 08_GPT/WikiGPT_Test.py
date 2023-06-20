import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# 读取对话数据
from Utilities import read_data
from CorpusLoader import WikiCorpus
corpus = WikiCorpus(read_data('01_Data/wikitext-103/wiki.train.txt'))  
vocab_size = len(corpus.vocab)

# 加载模型
import torch
from GPT_Model_with_Decode import GPT
device = "cuda" if torch.cuda.is_available() else "cpu"
loaded_model = GPT(corpus).to(device)
loaded_model.load_state_dict(torch.load('99_TrainedModel/WikiGPT_0.01_200_20230620_111344.pth'))
loaded_model.eval()  # 将模型设置为评估模式（推理模式）

# 测试模型 - Text #1
input_str = "let us go"
greedy_search_output = loaded_model.decode(input_str, strategy='greedy', max_len=25)
beam_search_output = loaded_model.decode(input_str, strategy='beam_search', 
                                         max_len=25, beam_width=5, repetition_penalty=1.2)
print("Input text:", input_str)
print("Greedy search output:", greedy_search_output)
print("Beam search output:", beam_search_output)

# 测试模型 - Text #2
input_str = "please tell me"
greedy_search_output = loaded_model.decode(input_str, strategy='greedy', max_len=25)
beam_search_output = loaded_model.decode(input_str, strategy='beam_search', 
                                         max_len=25, beam_width=5, repetition_penalty=1.2)
print("Input text:", input_str)
print("Greedy search output:", greedy_search_output)
print("Beam search output:", beam_search_output)

# 测试模型 - Text #3
input_str = "whoever knows this"
greedy_search_output = loaded_model.decode(input_str, strategy='greedy', max_len=25)
beam_search_output = loaded_model.decode(input_str, strategy='beam_search', 
                                         max_len=25, beam_width=5, repetition_penalty=1.2)
print("Input text:", input_str)
print("Greedy search output:", greedy_search_output)
print("Beam search output:", beam_search_output)

# 测试模型 - Text #4
input_str = "I can not guess"
greedy_search_output = loaded_model.decode(input_str, strategy='greedy', max_len=25)
beam_search_output = loaded_model.decode(input_str, strategy='beam_search', 
                                         max_len=25, beam_width=5, repetition_penalty=1.2)
print("Input text:", input_str)
print("Greedy search output:", greedy_search_output)
print("Beam search output:", beam_search_output)