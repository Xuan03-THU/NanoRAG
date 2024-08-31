# NanoRAG

## 介绍

NanoRAG 是一个本地运行的轻量化RAG模型，可以从本地文档生成向量库，并通过向量库进行相似句子检索。  
该模型使用尽量简单的架构，只保留最基本的功能。  
向量库无需训练。

## 快速开始

### 环境准备
1. 克隆存储库到本地: 
    ```
    git clone https://github.com/Xuan03-THU/NanoRAG.git
    ```
2. 安装依赖: 
    ```
    pip install -r requirements.txt
    ```

3. 下载预训练模型:   
   默认使用[bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5/tree/main)，把所有文件保存到子文件夹 './bge-small-zh-v1.5' 下。  
   也可以在初始化Retriever时指定路径。  
    还可以直接在初始化时通过huggingface下载:
    ```
    retriever = Retriever(embeddings_model='BAAI/bge-small-zh-v1.5')
    ```

### 使用NanoRAG
[demo.ipynb](https://github.com/Xuan03-THU/NanoRAG/blob/master/demo.ipynb)