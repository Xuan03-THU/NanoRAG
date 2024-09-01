'''
VectorStore: 基于FAISS的向量存储
Version: 0.1

@Author: 李语轩
@GitHub: https://github.com/Xuan03-THU

2024/8/31
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
        get_all_docs: 获取所有文档
        add: 添加文档
        delete: 删除文档
        delete_through_ids: 通过id删除文档
        search: 搜索文档
    '''

    def __init__(self, local_path=None, embeddings_model="./bge-small-zh-v1.5", nlist=1000):
        '''
        初始化: 新建一个存储器，或者从本地加载一个存储器

        参数:
            local_path: 本地存储器路径
            embeddings_model: 词向量模型
            nlist: 聚类中心数
        '''
        self.embeddings_model = embeddings_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        self.embeddings_dim = len(self.embeddings.embed_query("hello world"))
        
        if local_path is not None:
            self.vector_store = FAISS.load_local(local_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            # https://blog.csdn.net/ResumeProject/article/details/135350945
            # 下面两行的方法可能报错，因为需要训练
            # self.__quantizer = faiss.IndexFlatL2(self.embeddings_dim)
            # self.index = faiss.IndexIVFFlat(self.__quantizer, self.embeddings_dim, nlist)
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

    # TODO
    def merge(self, other_vector_store: 'VectorStore'):
        '''
        将另一个存储器合并到当前存储器

        参数:
            other_vector_store: 另一个存储器

        异常: 
            ValueError: 另一个存储器的词向量模型或维度不匹配
        '''
        # TODO: 合并时，是否需要重新训练索引？判断embeddings 是否一致的方法是否需要修改？
        # TODO: 合并文件列表
        if self.cache_folder!= other_vector_store.cache_folder or self.embeddings_dim!= other_vector_store.embeddings_dim:
            raise ValueError("Cannot merge two vector stores with different embeddings")
        self.vector_store.merge_from(other_vector_store.vector_store)

    def __get_ids(self, idx: int | list[int] | str | list[str]):
        '''
        获取id（如果是int 说明是索引，如果是str 说明是id）
        参考: https://github.com/langchain-ai/langchain/issues/8897

        参数:
            idx: 索引或id（列表）

        返回:
            id: id列表
        '''
        if isinstance(idx, int) or isinstance(idx, str):
            idx = [idx]
        
        ids = []
        for i in idx:
            if isinstance(i, str):
                if i not in self.vector_store.docstore._dict:
                    raise ValueError(f"Document with id {i} is not in the vector store")
                ids.append(i)
            else:
                if i not in self.vector_store.index_to_docstore_id:
                    raise ValueError(f"Index {i} is not in the vector store")
                ids.append(self.vector_store.index_to_docstore_id[i])
        return ids
    
    def get_docs(self, idx: int | list[int] | str | list[str]):
        '''
        获取文档

        参数:
            idx: 索引或id（列表）

        返回:
            docs: 文档列表
        '''
        ids = self.__get_ids(idx)
        return self.get_docs_through_ids(ids)
    
    def get_all_docs(self):
        '''
        获取所有文档

        返回:
            docs: 文档列表
        '''
        # Python 3.7+ only, 否则无法保证顺序
        return self.get_docs(list(self.vector_store.index_to_docstore_id.values()))

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

    def delete(self, idx: int | list[int] | str | list[str]):
        '''
        根据索引或id删除文档（新版本中，因为使用SHA256作为id，数据类型为str，所以可以进行区分）

        参数:
            idx: 索引（列表）

        异常:
            ValueError: 文档不存在
        '''
        ids = self.__get_ids(idx)

        for i in ids:       # 其实__get_ids已经保证了不存在的文档不会被删除
            if i not in self.vector_store.docstore._dict:   
                raise ValueError(f"Document with id {i} is not in the vector store")
        self.vector_store.delete(ids=ids)

    def delete_through_document(self, doc: Document):
        '''
        通过文档删除文档

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
