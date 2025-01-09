# Movie Assistant Chatbot

This project is a **Movie Assistant Chatbot** built using Python, Streamlit, and HuggingFace's Transformers library. The chatbot allows users to ask questions about movies from the IMDb Top 1000 dataset and provides answers by leveraging a Retrieval-Augmented Generation (RAG) pipeline.

---

## Features

1. **Interactive Chatbot Interface**:
   - Users can ask movie-related questions in real-time.
   - The chatbot responds with answers based on the provided IMDb Top 1000 dataset.

2. **Data Preprocessing**:
   - Combines relevant columns (e.g., title, plot, actors, director) into a single field for text embeddings.
   - Cleans and prepares data for efficient querying.

3. **Retrieval-Augmented Generation (RAG)**:
   - Uses a combination of HuggingFace embeddings and FAISS for efficient retrieval.
   - Employs the Flan-T5 model for generating high-quality responses.

4. **Save Chat History**:
   - Chat history is saved as `.txt`, `.pdf`, and `.xlsx` files for later reference.

---

## Requirements

### Python Libraries

Install the required libraries using the following command:

```bash
pip install streamlit pandas transformers langchain sentence-transformers fpdf
```

### File Requirements

- `imdb_top_1000.csv`: A dataset containing IMDb's top 1000 movies. The file should include columns such as `Title`, `Year`, `Plot`, `Director`, `Star1`, `Star2`, `Star3`, and `Star4`.

---

## How to Run

1. **Set Up the Environment**:
   - Ensure all required Python libraries are installed.
   - Place the `imdb_top_1000.csv` file in the same directory as the script.

2. **Run the Streamlit App**:
   - Save the script as `app.py`.
   - Start the Streamlit server:

     ```bash
     streamlit run app.py
     ```

   - Open the provided URL (usually `http://localhost:8501`) in a web browser.

3. **Chat with the Bot**:
   - Ask questions such as:
     - *"What is the plot of The Shawshank Redemption?"
     - *"Who directed Inception?"
   - View the chatbot's responses in real-time.

---

## Saving Chat History

The chatbot automatically saves chat history after the session ends in the following formats:

1. **TXT**: Plain text format.
2. **PDF**: Printable PDF format (requires `fpdf` library).
3. **XLSX**: Excel spreadsheet format.

The saved files will be named:

- `chatbot_responses.txt`
- `chatbot_responses.pdf`
- `chatbot_responses.xlsx`

---

## Code Structure

### Main Components

1. **Data Preprocessing**:
   - Merges actor columns into a single field.
   - Combines relevant movie information into a "Combined" column for embeddings.

2. **FlanT5LLM Class**:
   - A wrapper for the Flan-T5 model from HuggingFace.
   - Handles tokenization, generation, and response formatting.

3. **Setup RAG Pipeline**:
   - Creates embeddings using `sentence-transformers`.
   - Builds a FAISS vector store for efficient similarity search.
   - Configures a RetrievalQA pipeline using the Flan-T5 model.

4. **Chatbot Interface**:
   - Provides an interactive interface for users to ask questions.
   - Saves chat history for later reference.

---

## Example Usage

### Sample Questions

- "What is the plot of The Dark Knight?"
- "Who directed The Godfather?"
- "Can you list the main actors in Pulp Fiction?"
- "What is the genre of 12 Angry Men?"

---

## Troubleshooting

1. **File Not Found Error**:
   - Ensure the `imdb_top_1000.csv` file is in the correct directory.

2. **Library Import Errors**:
   - Ensure all required libraries are installed using `pip install`.

3. **FAISS Index Error**:
   - Verify that the dataset is correctly preprocessed and the "Combined" column exists.

---

## Credits

- **IMDb Dataset**: Sourced from IMDb Top 1000 movies.
- **HuggingFace Transformers**: Used for the Flan-T5 model.
- **LangChain**: For setting up the RAG pipeline.
- **Streamlit**: For creating the interactive web interface.

---

## License

This project is licensed under the MIT License.
