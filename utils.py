'''
utils: 工具函数
Version: 0.1

@Author: 李语轩
@GitHub: https://github.com/Xuan03-THU

2024/8/31
'''

from langchain.document_loaders import  PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import json

def read_pdf(path, chunk_size=1000, chunk_overlap=200):
    '''
    读取PDF文件，并将其分割为多个chunk，返回一个Document列表

    参数: 
        path: PDF文件路径
        chunk_size: 每个chunk的大小，默认为1000
        chunk_overlap: 两个chunk的重叠大小，默认为200  

    返回: 
        texts: 一个Document列表，每个Document包含一个chunk的文本，以及该chunk的元数据
    
    '''
    # Load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    for text in texts:
        text.page_content = text.page_content.replace("/t", " ")    # Replace tabs with spaces
        text.metadata["datatype"] = "str"

    return texts

def read_txt(path, chunk_size=1000, chunk_overlap=200):
    '''
    读取文本文件，并将其分割为多个chunk，返回一个Document列表

    参数: 
        path: 文本文件路径
        chunk_size: 每个chunk的大小，默认为1000
        chunk_overlap: 两个chunk的重叠大小，默认为200  

    返回: 
        texts: 一个Document列表，每个Document包含一个chunk的文本，以及该chunk的元数据
    
    '''
    # Load text files
    with open(path, "r") as f:
        content = f.read()
    
    texts = [Document(page_content=content, metadata={"source": path, "datatype": "str"})]

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(texts)
    for text in texts:
        text.page_content = text.page_content.replace("/t", " ")    # Replace tabs with spaces

    return texts

def encode(data) -> Document:
    '''
    将Python对象编码为Document对象

    参数: 
        data: Python对象，支持str、list、dict 等类型

    返回: 
        document: Document对象

    异常: 
        TypeError: 序列化失败
    '''
    if isinstance(data, str):
        return Document(page_content=data, metadata={"datatype": "str"})
    else:
        try:
            data_str = json.dumps(data)
            return Document(page_content=data_str, metadata={"source": None, "datatype": type(data).__name__})
        except TypeError as e:
            print(f"Error during serialization: {e}")
            return None

def decode(document: Document):
    '''
    将Document对象解码为Python对象

    参数: 
        document: Document对象

    返回: 
        data: Python对象 

    异常: 
        json.JSONDecodeError: 解码失败
    '''
    content, metadata = document.page_content, document.metadata
    if metadata["datatype"] == "str":
        return content
    else:
        try:
            data = json.loads(content)
            return data
        except json.JSONDecodeError as e:
            print(f"Error during deserialization: {e}")
            return None

