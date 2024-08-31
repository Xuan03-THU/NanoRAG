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

        self.index_to_docstore_id = self.vector_store.index_to_docstore_id
        self.docstore = self.vector_store.docstore

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
        if self.embeddings_model!= other_vector_store.embeddings_model or self.embeddings_dim!= other_vector_store.embeddings_dim:
            raise ValueError("Cannot merge two vector stores with different embeddings")
        self.vector_store.merge_from(other_vector_store.vector_store)

    def __get_ids(self, idx: int | list[int]):
        '''
        从索引获取id
        参考: https://github.com/langchain-ai/langchain/issues/8897

        参数:
            idx: 索引（列表）

        返回:
            id: id列表
        '''
        if isinstance(idx, int):
            idx = [idx]
        
        ids = []
        for i in idx:
            if i not in self.index_to_docstore_id:
                raise ValueError(f"Index {i} is not in the vector store")
            ids.append(self.index_to_docstore_id[i])
        return ids
    
    def get_docs(self, idx: int | list[int]):
        '''
        获取文档

        参数:
            idx: 索引（列表）

        返回:
            docs: 文档列表
        '''
        ids = self.__get_ids(idx)
        return self.get_docs_through_ids(ids)
        
    def get_docs_through_ids(self, ids: int | list[int]):
        '''
        通过id获取文档

        参数:
            ids: id（列表）

        返回:
            docs: 文档列表
        '''
        if isinstance(ids, int):
            ids = [ids]

        docs = []
        for i in ids:
            if i not in self.docstore._dict:
                raise ValueError(f"Document with id {i} is not in the vector store")
            docs.append(self.docstore._dict[i])
        return docs
    
    def get_all_docs(self):
        '''
        获取所有文档

        返回:
            docs: 文档列表
        '''
        # Python 3.7+ only, 否则无法保证顺序
        return self.get_docs(list(self.index_to_docstore_id.values()))

    def __len__(self):
        '''
        获取文档数量

        返回:
            文档数量
        '''
        return len(self.index_to_docstore_id)
    
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
            h = hash(doc.page_content)
            if h in self.docstore._dict:
                documents.remove(doc)
            else:
                ids.append(h)
        self.vector_store.add_documents(documents=documents, ids=ids)

    def delete(self, idx: int | list[int]):
        '''
        根据索引删除文档

        参数:
            idx: 索引（列表）

        异常:
            ValueError: 文档不存在
        '''
        ids = self.__get_ids(idx)
        self.delete_through_ids(ids)

    def delete_through_ids(self, ids: int | list[int]):
        '''
        通过id删除文档（因为id也是int，所以不能跟上面的delete方法合并）

        参数:
            ids: id（列表）

        异常:
            ValueError: 文档不存在
        '''
        if isinstance(ids, int):
            ids = [ids]
        for i in ids:
            if i not in self.docstore._dict:
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
        ids = [hash(doc.page_content)]
        self.delete_through_ids(ids)

    def search(self, query: str, top_k: int = 5):
        '''
        搜索文档

        参数:
            query: 查询语句
            top_k: 返回的文档数量

        返回:
            docs: 文档列表
        '''
        retriever_ = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        docs = retriever_.invoke(query)

        return docs
