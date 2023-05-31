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