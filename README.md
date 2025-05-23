# Akaike Interview Task

## Task

### LLM Powered Q/A System

Intelligent document assistant that can pull relevant answers from the text

> I chose the ACKO insurance document from the provided datasets

## Approach

1. **Data Preprocessing**:

   - Read the document and split it into smaller chunks.

2. **Embedding**:

   - Used Hugging Face's `sentence-transformers/all-minilm` to create embeddings for the chunks.

3. **Vector Store**:

   - Used `faiss` to create a vector store for efficient similarity search.

4. **Querying**:

   - For a given question, created an embedding and search the vector store for the most similar chunk.
   - Return the top 3 most relevant chunks as answers.

5. **LLM Integration**:
   - Used LLaMa 3.1 model from Groq provider to generate a final answer based on the retrieved chunks.
   - The LLM is prompted with the question and the top 3 chunks to provide a concise answer.

## Run

1. Clone the repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run
   ```bash
   streamlit run streamlit_app.py
   ```
