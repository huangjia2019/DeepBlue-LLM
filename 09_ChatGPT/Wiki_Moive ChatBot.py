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
from CorpusLoader import WikiCorpus
from Utilities import read_data, remove_input_from_output

# 加载 WikiCorpus 数据 
corpus = WikiCorpus(read_data('01_Data/wikitext-103/wiki.train.txt')) 

# 加载微调后的模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model_save_path = '99_TrainedModel/miniChatGPT_0.001_20000_20230609_231224.pth'
wiki_gpt = GPT(corpus).to(device)
wiki_gpt.load_state_dict(torch.load(model_save_path))
wiki_gpt.eval()

print("ChatGPT说: 您好，我们终于见面了！您希望接下来我为您提供什么服务？")
input("人类说：")
print("\n" , "ChatGPT说: 好的，明白了! 我会服务好您的。我的参数和语料库比较少,可能胡言乱语, 您多担待。" + 
      "\n" + "现在请和我聊天吧！" + "\n" + "记住，烦我的时候，请说‘再见’。")

while True:
    print("\n")
    message = input("人类说：")

    if message.lower() == "再见":
        print("ChatGPT说: 再见！很高兴为您服务，期待下次与您交流！")
        break

    output_text = wiki_gpt.decode(message, strategy='beam_search', 
                               max_len=20, beam_width=5, repetition_penalty=1.2)     
    reply = remove_input_from_output(message, output_text.strip()) if output_text is not None else '请输入有意义的句子。'
    print("ChatGPT说: ", reply)
