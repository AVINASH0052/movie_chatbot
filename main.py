import pandas as pd
import streamlit as st
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

# Load and preprocess data
def preprocess_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Data Loaded Successfully")
    except FileNotFoundError:
        print("File not found. Try Again!")
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
    print("Generating embeddings...")
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store
    print("Creating FAISS vector store...")
    vector_store = FAISS.from_texts(data['Combined'].tolist(), hf_embeddings)

    # Set up the Flan-T5 LLM
    print("Setting up Flan-T5 LLM...")
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
    print("Configuring RetrievalQA pipeline...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})  # Retrieve the top 1 document
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=flan_t5_llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        input_key="question",
    )
    print("RAG pipeline setup complete.")
    return retrieval_chain

def save_responses_to_files(questions, responses):
    # Save to .txt
    with open("chatbot_responses.txt", "w") as txt_file:
        for q, r in zip(questions, responses):
            txt_file.write(f"Question: {q}\nAnswer: {r}\n\n")

    # Save to .pdf
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        for q, r in zip(questions, responses):
            pdf.multi_cell(0, 10, f"Question: {q}\nAnswer: {r}\n\n")

        pdf.output("chatbot_responses.pdf")
    except ImportError:
        print("Install fpdf to export responses as a PDF.")

    # Save to .xlsx
    df = pd.DataFrame({"Question": questions, "Answer": responses})
    df.to_excel("chatbot_responses.xlsx", index=False)

    print("Responses saved to chatbot_responses.txt, chatbot_responses.pdf, and chatbot_responses.xlsx.")


def chatbot_interface(qa_pipeline):
    print("\nWelcome to the Chatbot! Type your question below (type 'exit' to quit):\n")

    questions = []
    responses = []

    while True:
        user_question = input("You: ")
        if user_question.lower() == "exit":
            print("Exiting chat. Goodbye!")
            break

        response = qa_pipeline.invoke({"question": user_question})
        answer = response.get("result", "Sorry, I couldn't find an answer.")
        print(f"Chatbot: {answer}\n")

        # Save questions and responses
        questions.append(user_question)
        responses.append(answer)

    # Export to files
    save_responses_to_files(questions, responses)

# Main program
data = preprocess_data(file_path)
if data is not None:
    qa_pipeline = setup_rag_pipeline(data)
    chatbot_interface(qa_pipeline)
