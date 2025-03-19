import os
import openai
import numpy as np
import streamlit as st  # Import Streamlit for secrets management
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFChatbot:
    def __init__(self, folder_path):
        """
        Initialize the chatbot with the document folder path.
        Uses API key from Streamlit Cloud Secrets.
        """
        self.folder_path = folder_path
        self.api_key = st.secrets["RAJIV_OPENAI_API_KEY"]  # Use Streamlit Cloud Secrets
        self.openai_client = openai.OpenAI(api_key=self.api_key)

        # Load and preprocess documents
        self.docs, self.filenames = self.load_documents()
        self.vectorizer, self.doc_vectors = self.vectorize_documents()

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

    def vectorize_documents(self):
        """Converts documents into TF-IDF vectors."""
        vectorizer = TfidfVectorizer(stop_words="english")
        doc_vectors = vectorizer.fit_transform(self.docs)
        return vectorizer, doc_vectors

    def retrieve_relevant_docs(self, query, top_n=3):
        """Retrieves the most relevant documents using cosine similarity."""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        return "\n".join([self.docs[i] for i in top_indices])

    def generate_response(self, context, user_query):
        """Generates a response using OpenAI GPT based on retrieved context."""
        prompt = f"Context: {context}\n\nUser query: {user_query}\n\nProvide a concise answer based on the context."
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Weaker GPT model for focused RAG-style responses
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


# **Run the Chatbot in Streamlit**
st.title("TF-IDF Chatbot")

folder_path = "data/processed/tfidf"
chatbot_instance = TFIDFChatbot(folder_path)

user_query = st.text_input("Ask the chemistry chatbot a question:")
if user_query:
    response = chatbot_instance.chatbot(user_query)
    st.write("### Chatbot Response:")
    st.write(response)