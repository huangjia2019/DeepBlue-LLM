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
import torch
from GPT_Model_with_Decode import GPT
from CorpusLoader import MovieCorpus
from Utilities import read_data
from ModelTrainer import Trainer
import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载 MovieCorpus 数据  
dialog_corpus = MovieCorpus(read_data('01_Data/cornell movie-dialogs corpus/processed_movie_lines.txt'))
chat_gpt = GPT(dialog_corpus).to(device)
chat_gpt.load_state_dict(torch.load('99_TrainedModel/WikiGPT_0.01_200_20230620_111344.pth'))
# chat_gpt.eval()

# 微调 ChatGPT 模型
trainer = Trainer(chat_gpt, dialog_corpus, learning_rate=0.001, epochs=200)
trainer.train()

# 保存微调后的模型
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # 获取当前时戳
model_save_path = f'99_TrainedModel/miniChatGPT_{trainer.lr}_{trainer.epochs}_{timestamp}.pth'
torch.save(chat_gpt.state_dict(), model_save_path)

# 测试微调后的模型
input_str = "how are you ?"
greedy_output = chat_gpt.decode(input_str, strategy='greedy', max_len=50)
beam_search_output = chat_gpt.decode(input_str, strategy='beam_search', max_len=50, beam_width=5, repetition_penalty=1.2)

print("Input text:", input_str)
# print("Greedy search output:", greedy_output)
print("Beam search output:", beam_search_output)
