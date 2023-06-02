#----------------------------------------------------
# 生成式预训练语言模型：理论与实战
# 深蓝学院 课程 
# 课程链接：https://www.shenlanxueyuan.com/course/620
#
# 作者 **黄佳**
#----------------------------------------------------
# 读取数据
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().split("\n")
    return data


def remove_input_from_output(input_text, output_text):
    # If the output text starts with the input text, remove it
    if output_text.startswith(input_text):
        output_text = output_text[len(input_text):]
    return output_text