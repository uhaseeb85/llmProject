import os
from typing import List, Dict
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from langchain.embeddings.base import Embeddings

class SimpleEmbeddings(Embeddings):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.fitted = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.fitted:
            self.vectorizer.fit(texts)
            self.fitted = True
        embeddings = self.vectorizer.transform(texts).toarray()
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        if not self.fitted:
            raise ValueError("Vectorizer not fitted. Call embed_documents first.")
        embedding = self.vectorizer.transform([text]).toarray()[0]
        return embedding.tolist()

class PDFKnowledgeBase:
    def __init__(self, persist_directory: str = "knowledge_base"):
        self.persist_directory = persist_directory
        self.embeddings = SimpleEmbeddings()
        self.vectorstore = self.load_or_create_vectorstore()

    def load_or_create_vectorstore(self):
        if os.path.exists(self.persist_directory):
            return FAISS.load_local(self.persist_directory, self.embeddings)
        return FAISS.from_texts(["Initialize knowledge base"], embedding=self.embeddings)

    def add_pdf(self, pdf_path: str):
        text = self.extract_text_from_pdf(pdf_path)
        self.add_text_to_vectorstore(text)

    def add_text_to_vectorstore(self, text: str):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        self.vectorstore.add_texts(chunks)
        self.vectorstore.save_local(self.persist_directory)

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        with open(pdf_path, "rb") as file:
            pdf = PdfReader(file)
            return "".join(page.extract_text() for page in pdf.pages)

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

    def answer_question(self, question: str) -> str:
        response = self.qa_chain({"question": question})
        return response['answer']

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