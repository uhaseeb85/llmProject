import os
import json
from typing import List, Dict
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS  # Updated import
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from langchain.embeddings.base import Embeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import faiss

class SimpleEmbeddings(Embeddings):
    def __init__(self, dimension=100):
        self.vectorizer = TfidfVectorizer(max_features=dimension, stop_words="english")
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts or all(len(text.strip()) == 0 for text in texts):
            return [[0.0] * self.dimension]
        embeddings = self.vectorizer.fit_transform(texts).toarray()
        return [self._pad_or_truncate(e) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        if not text or len(text.strip()) == 0:
            return [0.0] * self.dimension
        embedding = self.vectorizer.transform([text]).toarray()[0]
        return self._pad_or_truncate(embedding)

    def _pad_or_truncate(self, embedding: np.ndarray) -> List[float]:
        if embedding.shape[0] > self.dimension:
            return embedding[:self.dimension].tolist()
        elif embedding.shape[0] < self.dimension:
            return np.pad(embedding, (0, self.dimension - embedding.shape[0]), 'constant').tolist()
        return embedding.tolist()

class PDFKnowledgeBase:
    def __init__(self, persist_directory: str = "knowledge_base", dimension=100):
        self.persist_directory = persist_directory
        self.dimension = dimension
        self.embeddings = SimpleEmbeddings(dimension=self.dimension)
        self.ensure_directory_exists()
        self.vectorstore = self.load_or_create_vectorstore()
        self.processed_pdfs = self.load_processed_pdfs()

    def ensure_directory_exists(self):
        os.makedirs(self.persist_directory, exist_ok=True)

    def load_or_create_vectorstore(self):
        if os.path.exists(os.path.join(self.persist_directory, "index.faiss")):
            return FAISS.load_local(
                self.persist_directory, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            # Create a new FAISS instance with a dummy text
            vectorstore = FAISS.from_texts(["dummy text"], self.embeddings)
            vectorstore.save_local(self.persist_directory)
            return vectorstore

    def load_processed_pdfs(self):
        pdf_list_file = os.path.join(self.persist_directory, "processed_pdfs.json")
        if os.path.exists(pdf_list_file):
            with open(pdf_list_file, 'r') as f:
                return json.load(f)
        else:
            return []

    def save_processed_pdfs(self):
        pdf_list_file = os.path.join(self.persist_directory, "processed_pdfs.json")
        with open(pdf_list_file, 'w') as f:
            json.dump(self.processed_pdfs, f)

    def add_pdf(self, file_path: str):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(pages)
        
        # Extract text content from Document objects
        text_content = [doc.page_content for doc in texts]
        
        self.vectorstore.add_texts(text_content)
        self.processed_pdfs.append(file_path)
        self.save_processed_pdfs()
        self.vectorstore.save_local(self.persist_directory)

    def get_processed_pdfs(self):
        return self.processed_pdfs

class PDFQuestionAnswering:
    def __init__(self, knowledge_base: PDFKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.llm = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="not-needed",
            model_name="local-model"
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.knowledge_base.vectorstore.as_retriever(),
            memory=self.memory
        )

    def answer_question(self, question: str, update_function=None):
        handler = StreamingHandler(update_function) if update_function else None
        response = self.qa_chain.invoke(
            {"question": question},
            config={"callbacks": [handler]} if handler else {}
        )
        return response['answer']

class StreamingHandler(BaseCallbackHandler):
    def __init__(self, update_function):
        self.update_function = update_function

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.update_function:
            self.update_function(token)

def main():
    kb = PDFKnowledgeBase()
    qa = PDFQuestionAnswering(kb)

    while True:
        action = input("Enter 'add' to add a PDF, 'ask' to ask a question, or 'quit' to exit: ")
        if action.lower() == 'add':
            pdf_path = input("Enter the path to the PDF file: ")
            kb.add_pdf(pdf_path)
            print("PDF added to knowledge base.")
        elif action.lower() == 'ask':
            question = input("Enter your question: ")
            answer = qa.answer_question(question)
            print(f"Answer: {answer}")
        elif action.lower() == 'quit':
            break
        else:
            print("Invalid action. Please try again.")

if __name__ == "__main__":
    main()