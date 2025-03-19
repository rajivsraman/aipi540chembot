import os
import openai
import faiss
import numpy as np
import streamlit as st
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class Doc2VecChatbot:
    def __init__(self, folder_path, use_pretrained=False):
        """
        Initialize the chatbot with Doc2Vec-based document embeddings stored in FAISS.
        Uses API key from Streamlit Cloud Secrets.
        """
        self.folder_path = folder_path
        self.api_key = os.getenv("RAJIV_OPENAI_API_KEY")  # Use Streamlit Secrets in app.py
        self.openai_client = openai.OpenAI(api_key=self.api_key)

        # Load and preprocess documents
        self.docs, self.filenames = self.load_documents()

        # Train or load Doc2Vec model
        self.model = self.train_doc2vec()
        self.vector_size = self.model.vector_size

        # Create FAISS index
        self.index, self.doc_mapping = self.create_faiss_index()

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

    def train_doc2vec(self):
        """Trains a Doc2Vec model using the text documents."""
        tagged_docs = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(self.docs)]
        model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=20)
        model.build_vocab(tagged_docs)
        model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
        return model

    def get_embedding(self, doc_index):
        """Gets the document embedding from the trained Doc2Vec model."""
        return self.model.dv[doc_index]

    def create_faiss_index(self):
        """Stores Doc2Vec embeddings in FAISS with cosine similarity."""
        d = self.vector_size  # Dimension of embeddings
        index = faiss.IndexFlatIP(d)  # Inner Product index (cosine similarity)

        embeddings = []
        doc_mapping = {}

        for i in range(len(self.docs)):
            vector = self.get_embedding(i)
            embeddings.append(vector)
            doc_mapping[i] = self.docs[i]  # Store mapping of index to document

        # Convert to numpy array and add to FAISS index
        embeddings = np.array(embeddings).astype('float32')
        index.add(embeddings)

        return index, doc_mapping

    def retrieve_relevant_docs(self, query, top_n=3):
        """Retrieves the most relevant documents from FAISS using cosine similarity."""
        query_embedding = np.array([self.model.infer_vector(query.split())]).astype('float32')
        _, indices = self.index.search(query_embedding, top_n)

        return "\n".join([self.doc_mapping[i] for i in indices[0] if i in self.doc_mapping]) if len(indices[0]) > 0 else "No relevant documents found."

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
