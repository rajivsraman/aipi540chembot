import os
import openai
import faiss
import numpy as np
import streamlit as st
from gensim.models import Word2Vec

class Word2VecChatbot:
    def __init__(self, folder_path):
        """
        Initialize the chatbot with Word2Vec-based document embeddings stored in FAISS.
        Uses API key from Streamlit Cloud Secrets.
        """
        self.folder_path = folder_path
        self.api_key = os.getenv("RAJIV_OPENAI_API_KEY")  # Use Streamlit Secrets in app.py
        self.openai_client = openai.OpenAI(api_key=self.api_key)

        # Load and preprocess documents
        self.docs, self.filenames = self.load_documents()
        self.model = self.train_word2vec()
        self.index = self.create_faiss_index()  # Use FAISS instead of ChromaDB

    def load_documents(self):
        """Loads text documents from the specified folder."""
        docs, filenames = [], []
        for file in os.listdir(self.folder_path):
            if file.endswith(".txt"):
                with open(os.path.join(self.folder_path, file), "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    docs.append(content)
                    filenames.append(file.replace(".txt", ""))
        return docs, filenames

    def train_word2vec(self):
        """Trains a Word2Vec model using the text documents."""
        tokenized_docs = [doc.split() for doc in self.docs]
        return Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

    def get_embedding(self, text):
        """Converts text into a Word2Vec embedding by averaging word vectors."""
        words = text.split()
        vectors = [self.model.wv[word] for word in words if word in self.model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.model.vector_size)

    def create_faiss_index(self):
        """Stores Word2Vec embeddings in FAISS."""
        d = self.model.vector_size  # Dimension of embeddings
        index = faiss.IndexFlatL2(d)  # L2 distance FAISS index

        embeddings = []
        for doc in self.docs:
            vector = self.get_embedding(doc)
            embeddings.append(vector)
        
        # Convert to numpy array and add to FAISS index
        embeddings = np.array(embeddings).astype('float32')
        index.add(embeddings)

        return index

    def retrieve_relevant_docs(self, query, top_n=3):
        """Retrieves the most relevant documents from FAISS using similarity search."""
        query_embedding = np.array([self.get_embedding(query)]).astype('float32')
        _, indices = self.index.search(query_embedding, top_n)
        return "\n".join([self.docs[i] for i in indices[0]]) if indices[0].size > 0 else "No relevant documents found."

    def generate_response(self, context, user_query):
        """Generates a response using OpenAI GPT based on retrieved context."""
        prompt = f"Context: {context}\n\nUser query: {user_query}\n\nProvide a concise answer based on the context."
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a chemistry assistant. ONLY use the provided context to generate responses. If the context does not contain the answer, respond with 'I don't know based on the provided context.' Do NOT use any external knowledge."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def chatbot(self, user_query):
        """Main chatbot function to process user queries."""
        context = self.retrieve_relevant_docs(user_query)
        return self.generate_response(context, user_query)


