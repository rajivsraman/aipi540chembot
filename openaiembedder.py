import os
import faiss
import openai
import numpy as np
import streamlit as st

class OpenAIEmbeddingsChatbot:
    def __init__(self, folder_path):
        """
        Initializes the chatbot with OpenAI embeddings stored in FAISS.
        Uses API key from Streamlit Cloud Secrets.
        """
        self.folder_path = folder_path
        self.api_key = os.getenv("RAJIV_OPENAI_API_KEY")  # Use Streamlit Secrets
        openai.api_key = self.api_key

        # Load and preprocess documents
        self.docs, self.filenames = self.load_documents()
        self.embeddings = self.generate_embeddings()
        
        # Initialize FAISS Index
        self.index = self.store_embeddings()

    def load_documents(self):
        """Loads cleaned text documents from the specified folder."""
        docs, filenames = [], []
        for file in os.listdir(self.folder_path):
            if file.endswith(".txt"):
                with open(os.path.join(self.folder_path, file), "r", encoding="utf-8") as f:
                    content = f.read().strip().lower()
                    docs.append(content)
                    filenames.append(file.replace(".txt", ""))
        return docs, filenames

    def generate_embeddings(self):
        """Generates OpenAI text embeddings for each document."""
        response = openai.embeddings.create(
            input=self.docs,  
            model="text-embedding-ada-002"
        )
        embeddings = [np.array(embedding.embedding) for embedding in response.data]
        return embeddings

    def store_embeddings(self):
        """Stores OpenAI-generated embeddings in FAISS."""
        dimension = len(self.embeddings[0])  # OpenAI embeddings are 1536-dimensional
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(self.embeddings, dtype=np.float32))
        return index

    def retrieve_relevant_docs(self, query, top_n=3):
        """Retrieves the most relevant documents using FAISS similarity search."""
        query_embedding = openai.embeddings.create(
            input=[query], 
            model="text-embedding-ada-002"
        ).data[0].embedding

        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_n)

        retrieved_texts = [self.docs[i] for i in indices[0] if i < len(self.docs)]
        return "\n".join(retrieved_texts) if retrieved_texts else "No relevant documents found."

    def generate_response(self, context, user_query):
        """Generates a response using OpenAI GPT based on retrieved context."""
        prompt = f"Context: {context}\n\nUser query: {user_query}\n\nProvide a concise answer based on the context."
        response = openai.chat.completions.create(
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