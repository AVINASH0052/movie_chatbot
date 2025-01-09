import streamlit as st
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import List, Optional
from pydantic import Field

# File Path
file_path = "imdb_top_1000.csv"

# Preprocess Data
@st.cache_data
def preprocess_data(file_path):
    try:
        data = pd.read_csv(file_path)
        st.write("Data Loaded Successfully")
    except FileNotFoundError:
        st.error("File not found. Try Again!")
        return None

    # Merge actor columns
    data['Actors'] = data['Star1'] + ', ' + data['Star2'] + ', ' + data['Star3'] + ', ' + data['Star4']
    data = data.drop(columns=['Star1', 'Star2', 'Star3', 'Star4', 'Poster_Link'])

    # Rename columns
    data.columns = ['Title', 'Year', 'Certificate', 'Duration', 'Genre', 'Rating', 'Plot',
                    'Score', 'Director', 'Votes', 'Gross', 'Actors']

    # Combine relevant columns into text for embeddings
    data['Combined'] = data.apply(
        lambda row: f"{row['Title']} ({row['Year']}): {row['Plot']} Director: {row['Director']} Actors: {row['Actors']}",
        axis=1
    )
    return data


class FlanT5LLM(LLM):
    model_name: str = "google/flan-t5-small"
    tokenizer: AutoTokenizer = Field(default=None, exclude=True)
    model: AutoModelForSeq2SeqLM = Field(default=None, exclude=True)

    def __init__(self, model_name: str = "google/flan-t5-small"):
        super().__init__()
        self.model_name = model_name

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    @property
    def _llm_type(self) -> str:
        return "flan-t5"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=250,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def setup_rag_pipeline(data):
    # Create embeddings
    st.write("Generating embeddings...")
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store
    st.write("Creating FAISS vector store...")
    vector_store = FAISS.from_texts(data['Combined'].tolist(), hf_embeddings)

    # Set up the Flan-T5 LLM
    st.write("Setting up Flan-T5 LLM...")
    flan_t5_llm = FlanT5LLM()

    # Define a refined prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an expert movie assistant. Use the provided context to answer the question accurately.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
    )

    # Set up RetrievalQA chain
    st.write("Configuring RetrievalQA pipeline...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})  # Retrieve the top 1 document
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=flan_t5_llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        input_key="question",
    )
    st.write("RAG pipeline setup complete.")
    return retrieval_chain


# Streamlit App
st.title("Movie Assistant Chatbot")
st.write("Ask any question about movies from the IMDb Top 1000 dataset!")

# Load data and pipeline
data = preprocess_data(file_path)

if data is not None:
    qa_pipeline = setup_rag_pipeline(data)

    # Chatbot Interface
    user_question = st.text_input("Ask a question (type and press Enter):")

    if user_question:
        response = qa_pipeline.invoke({"question": user_question})
        answer = response.get("result", "Sorry, I couldn't find an answer.")
        st.write(f"**Chatbot:** {answer}")

        # Save history
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].append((user_question, answer))

    # Display chat history
    if "history" in st.session_state and st.session_state["history"]:
        st.write("### Chat History:")
        for i, (q, a) in enumerate(st.session_state["history"], 1):
            st.write(f"**Q{i}:** {q}")
            st.write(f"**A{i}:** {a}")
