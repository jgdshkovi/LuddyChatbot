# 📚 LuddyBot Backend

This repository contains the backend implementation of LuddyBot, a context-aware AI assistant designed for Indiana University's Luddy School. It includes two main components: document embedding generation and chatbot response handling using LLaMA-based inference and hybrid search with reranking.

### 📁 Contents
- `createEmbeddings.py`: Script to load, chunk, embed, and index academic documents using FAISS.

- `chatbotV2.py`: FastAPI-based application that serves LuddyBot responses based on hybrid retrieval and LLM inference.

### ⚙️ Setup Instructions
#### 1. Environment Setup
Ensure you have Python 3.9+ and install the dependencies:

`pip install -r requirements.txt`


Dependencies include:
- langchain
- sentence-transformers
- faiss-cpu
- fastapi
- uvicorn
- groq
- pydantic
- matplotlib
- scikit-learn
- seaborn
- cohere
- tqdm


#### 2. Embedding Creation
Before running the chatbot, generate the FAISS index:

`python createEmbeddings.py`

- Place all .txt source documents inside a folder named Final_scraped_txts/.
- The script will generate and save the FAISS index at faiss_index_luddy/.

### 🤖 Running the Chatbot
Set the `GROQ_API_KEY` environment variable with your API key.
Set environment variable if using Cohere:
`export COHERE_API_KEY=your_api_key`


Then run the FastAPI server:
`uvicorn chatbotV2:app --reload`

Send POST requests to /ask endpoint with the following JSON format:

```
{  
    "prompt": "What are the Luddy career services hours?", 
    "session_id": "student123" 
}
```


### 🔍 Core Features
**Hybrid Search:** Combines semantic similarity with keyword filtering.

**Re-ranking:** Uses CrossEncoder (ms-marco-MiniLM-L6-v2) to prioritize relevant documents.

**LLaMA Inference:** Uses Groq's hosted LLaMA3-70B model for generating structured, context-sensitive answers.

**Session History:** Maintains conversation flow with SQLite-backed chat history.

**Contextual Summarization:** Automatically summarizes long conversations to maintain context continuity.





# SCRAPER

For the scraper to run, first run the `lvl0_scrape.py`, then the `lvl1` script and so on.
This way we organize the each level of scraping into folders and can later combine the files from the required no. of levels.

We are also keeping track of discovered links and visited links, to make sure we not visiting the webpages again and again, and make our scraping data redundant.



# RAG Evaluation Framework

This project provides a robust evaluation pipeline for Retrieval-Augmented Generation (RAG) systems. It computes both retrieval and generation quality metrics to assess performance in question-answering tasks.

### 📂 Contents
`rag-eval-enhanced.py`: Main script to evaluate RAG systems using FAISS, LLMs, rerankers, and various IR/NLP metrics.

Supports hybrid search, reranking, semantic similarity, and answer faithfulness scoring.


### ⚙️ Features
- Retrieval Metrics: Precision@K, Recall@K, MRR, NDCG

- Generation Metrics: Semantic Similarity, Faithfulness, Answer Relevance

- Hybrid Document Retrieval with FAISS and CrossEncoder/Cohere reranker

- LLM Answer Generation (e.g., LLaMA integration)

- Visualization of results using matplotlib


### 🚀 Usage

```
python rag-eval-enhanced.py \
  --test_data rag_test_data.json \
  --output rag-eval-enhanced-llama-8b/evaluation_results.csv \
  --detailed_output rag-eval-enhanced-llama-8b/evaluation_detailed.json \
  --plot rag-eval-enhanced-llama-8b/evaluation_plots.png \
  --top_n_relevant 3
```


### 🧪 Input Format
The script expects a JSON file like:

```
[
  {
    "question": "What is the capital of France?",
    "reference_answer": "Paris is the capital of France."
  }
]
```
To generate a sample template:

`python rag-eval-enhanced.py --test_data rag_test_data.json`

### 📈 Output
- `evaluation_results.csv`: Summary of metrics per query + average row
- `evaluation_detailed.json`: Full breakdown with individual scores
- `evaluation_plots.png`: Bar plots for each metric group


### 🧠 Customization
- Change model paths in the RAGEvaluator constructor
- Add support for other rerankers or embeddings
- Integrate with different LLMs for response generation

# For UI

### LuddyBuddy Chat Widget

A simple React chat widget that doubles as a Chrome extension.

#### Development

1. **Clone your repo** and enter the folder:
   ```bash
   git clone <your-repo-url>
   cd <app-folder> [UI]
   ```

2. **Install dependencies**:
   ```bash
   npm install        # or yarn install, or pnpm install
   ```
---

### Loading as a Chrome Extension

To install LuddyBuddy as a Chrome extension, follow these steps:

1. **Build the extension**
   ```bash
   pnpm run build      # or npm run build, yarn build
   ```

2. **Open Chrome’s Extensions page**
   - In your browser’s address bar, go to `chrome://extensions`.

3. **Enable Developer mode**
   - Toggle the **Developer mode** switch in the top-right corner of the page.

4. **Load the unpacked extension**
   - Click the **Load unpacked** button.
   - Select the root folder of your project (the one containing `manifest.json` and the `build` directory).

5. **Verify installation**
   - LuddyBuddy should now appear in your list of extensions and be ready to use.

