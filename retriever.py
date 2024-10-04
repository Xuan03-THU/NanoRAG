'''
Retriever: 利用VectorStore进行文档检索
Version: 0.1.1

@Author: 李语轩
@GitHub: https://github.com/Xuan03-THU

2024/10/04
'''


from vectorstore import VectorStore
from utils import *

import os
import json

class Retriever():
    '''
    Retriever: 利用VectorStore进行文档检索

    方法: 
        初始化: 新建一个Retriever对象，传入本地路径和向量模型路径用于构造VectorStore对象
        read: 把文件当作文字信息，并将其内容添加到VectorStore中
        load: 把文件当作数据列表，并将其添加到VectorStore中
        save: 保存VectorStore和文件列表
        retrieve: 根据query检索文档，返回top_k个结果
        remove: 删除某个文档
    '''

    def __init__(self, local_path=None, embeddings_model="./bge-small-zh-v1.5", 
                 device=None):
        '''
        初始化: 新建一个Retriever对象，传入本地路径和向量模型路径用于构造VectorStore对象

        参数: 
            local_path: 本地路径，用于保存向量模型和文件列表
            embeddings_model: 向量模型路径
            device: 模型的device
        '''
        self.vs = VectorStore(local_path=local_path, embeddings_model=embeddings_model, device=device)
        self.local_path = local_path
        if local_path is not None:
            try:
                with open(os.path.join(local_path, "files.json"), 'r') as f:
                    self.files = set(json.load(f))
            except:
                print("No files.json found, create a new one")
                self.files = set()
        else:
            self.files = set()

    def read(self, file_path, force_read=False):
        '''
        把文件当作文字信息，并将其内容添加到VectorStore中
        支持的格式: pdf, txt

        参数:
            file_path: 文件路径
            force_read: 是否强制读取文件，即使文件已经被读取过

        异常:
            ValueError: 不支持的文件类型
        '''
        if file_path in self.files and not force_read:
            print(f"{file_path} has been read before, skip")
            return
        suffix = file_path.split(".")[-1]
        if suffix == 'pdf':
            texts = read_pdf(file_path)
        elif suffix == 'txt':
            texts = read_txt(file_path)
        else:
            raise ValueError("Unsupported file type")
        
        self.vs.add(texts)
        self.files.add(file_path)

    def load(self, file_path, force_read=False):
        '''
        把文件当作数据列表，并将其添加到VectorStore中
        支持的格式: json, txt

        参数:
            file_path: 文件路径
            force_read: 是否强制读取文件，即使文件已经被读取过

        异常:
            ValueError: 不支持的文件类型
        '''
        if file_path in self.files and not force_read:
            print(f"{file_path} has been loaded before, skip")
            return
        suffix = file_path.split(".")[-1]
        if suffix not in ['json', 'txt']:
            raise ValueError(f"Unsupported file type {suffix}")
        
        with open(file_path, 'rb') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Can only load from a list of data")
        
        texts = []
        for item in data:
            text = encode(item)
            if text is not None:
                texts.append(text)

        self.vs.add(texts)
        self.files.add(file_path)

    def save(self, local_path=None):
        '''
        保存VectorStore和文件列表

        参数:
            local_path: 本地路径，用于保存向量模型和文件列表

        异常:
            ValueError: 本地路径未指定
        '''
        if local_path is None:
            if self.local_path is None:
                raise ValueError("No local path specified")
            local_path = self.local_path
        self.vs.save(local_path)
        with open(os.path.join(local_path, "files.json"), 'w') as f:
            json.dump(list(self.files), f)

    def retrieve(self, query, top_k=5, get_scores=False):
        '''
        根据query检索文档，返回top_k个结果

        参数:
            query: 查询语句
            top_k: 返回的结果个数
            get_scores: 是否返回分数

        返回:
            一个列表，包含top_k个结果的文本（或文本和分数）
        '''
        res = self.vs.search(query, top_k, get_scores=get_scores)

        contents = []
        if not get_scores:
            for item in res:
                contents.append(decode(item))
        else:
            for item, score in res:
                contents.append((decode(item), score))
        return contents
        
    def remove(self, data):
        '''
        删除某个文档

        参数:
            data: 文档数据，可以是文本或字典

        返回:
            True: 删除成功
            False: 删除失败

        异常:
            ValueError: 文档不存在
        '''
        try:
            doc = encode(data)
        except:
            raise ValueError("Invalid data")
        try:
            self.vs.delete_document(doc)
            return True
        except Exception as e:
            print(e)
            return False
