'''
VectorStore: 基于FAISS的向量存储
Version: 0.1.1

@Author: 李语轩
@GitHub: https://github.com/Xuan03-THU

2024/10/04
'''

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

import faiss
from utils import *

# 需要添加当前目录到系统路径，保证HuggingFaceEmbeddings可以通过model_name加载本地模型
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.'))) 


class VectorStore():
    '''
    VectorStore: 基于FAISS的向量存储

    方法: 
        初始化: 新建一个存储器，或者从本地加载一个存储器
        save: 保存当前存储器到本地
        merge: 将另一个存储器合并到当前存储器
        get_docs: 获取文档
        get_all_docs: 获取所有文档
        __len__: 获取文档数量
        add: 添加文档
        delete: 按照id删除文档
        delete_document: 通过文档内容删除文档
        search: 搜索文档
    '''

    def __init__(self, local_path=None, embeddings_model=None, 
                 device=None):
        '''
        初始化: 新建一个存储器，或者从本地加载一个存储器

        参数:
            local_path: 本地存储器路径
            embeddings_model: 词向量模型
            device: 模型的device
        '''
        self.embeddings_model = embeddings_model
        self.device = device if device is not None else 'cpu'
        model_kwargs = {'device': self.device}
        # encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model, 
                        model_kwargs=model_kwargs)
        self.embeddings_dim = len(self.embeddings.embed_query("hello world"))
        
        if local_path is not None:
            self.vector_store = FAISS.load_local(local_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            self.index = faiss.IndexFlatL2(self.embeddings_dim)
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=self.index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

    def save(self, local_path='./vector_store.faiss'):
        '''
        保存当前存储器到本地

        参数:
            local_path: 本地存储器路径
        '''
        self.vector_store.save_local(local_path)

    def merge(self, other_vector_store: 'VectorStore'):
        '''
        将另一个存储器合并到当前存储器

        参数:
            other_vector_store: 另一个存储器

        异常: 
            ValueError: 另一个存储器的词向量模型或维度不匹配
        '''
        # TODO: 合并时，是否需要重新训练索引？判断embeddings 是否一致的方法是否需要修改？
        if self.embeddings_model!= other_vector_store.embeddings_model \
            or self.embeddings_dim!= other_vector_store.embeddings_dim:
            raise ValueError("Cannot merge two vector stores with different embeddings")
        self.vector_store.merge_from(other_vector_store.vector_store)

    def get_docs(self, ids: str | list[str]):
        '''
        获取文档

        参数:
            ids: id（列表）

        返回:
            docs: 文档列表
        '''
        docs = []
        for i in ids:
            if i not in self.vector_store.docstore._dict:
                raise ValueError(f"Document with id {i} is not in the vector store")
            docs.append(self.vector_store.docstore._dict[i])
        return docs
    
    def get_all_docs(self):
        '''
        获取所有文档

        返回:
            docs: 文档列表
        '''
        return self.get_docs(list(self.vector_store.docstore._dict.keys()))

    def __len__(self):
        '''
        获取文档数量

        返回:
            文档数量
        '''
        return len(self.vector_store.index_to_docstore_id)
    
    def add(self, documents: Document | list[Document]):
        '''
        添加文档

        参数:
            documents: 文档（列表）

        注意: 
            文档的id会被设置为hash值，如果文档已经存在，则不会被添加
        ''' 
        if isinstance(documents, Document):
            documents = [documents]
        # 用hash值作为id
        ids = []
        for doc in documents:
            h = get_hash(doc)
            if h in self.vector_store.docstore._dict:
                documents.remove(doc)
            else:
                ids.append(h)
        self.vector_store.add_documents(documents=documents, ids=ids)

    def delete(self, ids: str | list[str]):
        '''
        根据id删除文档（id 是文档内容的SHA256哈希值）

        参数:
            ids: id（列表）

        异常:
            ValueError: 文档不存在
        '''
        for i in ids:
            if i not in self.vector_store.docstore._dict:   
                raise ValueError(f"Document with id {i} is not in the vector store")
        self.vector_store.delete(ids=ids)

    def delete_document(self, doc: Document):
        '''
        通过文档内容删除文档

        参数:
            doc: 文档

        异常:
            ValueError: 文档不存在            
        '''
        ids = [get_hash(doc)]
        self.delete(ids)

    def search(self, query: str, top_k: int = 5, get_scores=False):
        '''
        搜索文档

        参数:
            query: 查询语句
            top_k: 返回的文档数量
            get_scores: 是否返回相关性得分

        返回:
            docs: 文档列表
        '''
        if get_scores:
            return self.vector_store.similarity_search_with_relevance_scores(query, k=top_k)
        else:
            return self.vector_store.similarity_search(query, k=top_k)
