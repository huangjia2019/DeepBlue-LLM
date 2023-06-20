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

from Utilities import read_data

# 导入数据集    
from CorpusLoader import WikiCorpus
corpus = WikiCorpus(read_data('01_Data/wikitext-103/wiki.train.txt')) 

# 导入GPT模型
from GPT_Model_with_Decode import GPT
WikiGPT = GPT(corpus)
print(WikiGPT) # 打印模型架构

# 训练GPT模型
from ModelTrainer import Trainer
trainer = Trainer(WikiGPT, corpus, learning_rate=0.01, epochs = 200)
trainer.train()  

import datetime
# 获取当前时间戳
now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")

# 保存模型
import torch
model_save_path = f'99_TrainedModel/WikiGPT_{trainer.lr}_{trainer.epochs}_{timestamp}.pth'
torch.save(WikiGPT.state_dict(), model_save_path)