import pickle

# 指定文件路径
file_path = r'E:\python_project\logbert-main\logbert-main\output\hdfs\vocab.pkl'

# 使用 pickle.load() 加载.pkl文件
with open(file_path, 'rb') as f:
    vocab = pickle.load(f)

# 查看加载的内容
print(vocab)
