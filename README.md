# 🛡️ Multi-PDF RAG Chatbot with Guardrails & Evaluation Metrics

An advanced **Retrieval-Augmented Generation (RAG)** application that allows users to upload multiple PDF documents and ask context-aware questions through an interactive chatbot.

Built using **LangChain, Google Gemini, FAISS, and Streamlit**, the system retrieves relevant document chunks, generates answers grounded in uploaded documents, applies input/output guardrails, and evaluates response quality using Information Retrieval (IR) and RAG evaluation metrics.

## 🚀 Key Features

- **Multi-PDF Upload:** Upload and process multiple PDF documents simultaneously.
- **RAG Pipeline:** Retrieve relevant document chunks using semantic similarity search.
- **Vector Database:** FAISS for efficient vector storage and retrieval.
- **LLM Integration:** Google Gemini Flash for context-aware answer generation.
- **Input Guardrails:** Detects prompt injection attempts and blocks suspicious queries.
- **Output Guardrails:** Detects hallucinations, toxicity, and sensitive information.
- **PII Detection & Masking:** Identifies and redacts sensitive information such as PAN, GST, credit card numbers, SSN, bank account numbers, and API keys.
- **Source Tracking:** Displays source document names, page numbers, and retrieved chunks.
- **RAG Evaluation:** Evaluates retrieval and generation quality.
- **Performance Monitoring:** Tracks latency, input tokens, output tokens, total tokens, and remaining context window.
- **Interactive Dashboard:** Streamlit-based interface for chatting and monitoring evaluation metrics.

## 🏗️ System Architecture

```text
              User
               |
               v
        Streamlit Interface
               |
       Upload Multiple PDFs
               |
               v
          PyPDFLoader
               |
               v
      Text Chunking
   (Chunk Size: 1000)
   (Overlap: 50)
               |
               v
      Gemini Embeddings
               |
               v
          FAISS Index
               |
               v
         User Query
               |
               v
       Input Guardrails
               |
               v
       Similarity Search
          (Top K = 4)
               |
               v
      Retrieved Context
               |
               v
       Gemini Flash LLM
               |
               v
       Generated Answer
               |
               v
       Output Guardrails
               |
        PII Redaction
               |
               v
       RAG Evaluation
               |
               v
   Metrics & Source Tracking
               |
               v
        Final Response
```

## 📊 Evaluation Metrics

The application evaluates retrieval performance, answer quality, and factual integrity.

| Metric | Description |
|---|---|
| Precision@K | Measures the proportion of retrieved chunks considered relevant. |
| Recall@K | Estimates retrieval coverage based on the evaluator's relevant chunk count. |
| Faithfulness | Measures how well the response is grounded in retrieved context. |
| Context Recall | Evaluates how much relevant information is captured from the context. |
| Answer Relevance | Measures how relevant the generated answer is to the query. |
| Correctness | Evaluates the correctness of the generated response. |
| Hallucination Rate | Estimates the proportion of unsupported factual claims. |

### ⚡ Performance Metrics

- Response Latency
- Input Tokens
- Output Tokens
- Total Tokens
- Maximum Context Window
- Remaining Context

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM | Google Gemini Flash |
| Framework | LangChain |
| Embeddings | Gemini Embedding |
| Vector Database | FAISS |
| PDF Processing | PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| Evaluation | LLM-as-a-Judge |
| Security | Regex, PII Masking, Prompt Injection Detection |

## 📂 Project Structure

```text
Multi-PDF-RAG-Chatbot/
│
├── app.py              # Main Streamlit application and RAG pipeline
├── evals.py            # Retrieval and generation evaluation
├── guardrails.py       # Input/output security and PII masking
├── requirements.txt    # Project dependencies
├── .env                # API configuration (not committed)
└── README.md
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Multi-PDF-RAG-Chatbot.git
```

### 2. Navigate to Project Directory

```bash
cd Multi-PDF-RAG-Chatbot
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Gemini API Key

Get your API key from [Google AI Studio](https://aistudio.google.com/).

You can enter the API key directly in the Streamlit sidebar when running the application.

### 6. Run the Application

```bash
streamlit run app.py
```

## 🔐 Security Implementation

The project implements multiple security layers:

1. **Prompt Injection Detection:** Blocks queries containing predefined prompt injection patterns.
2. **Sensitive Data Detection:** Uses regular expressions to identify sensitive information.
3. **PII Masking:** Replaces detected sensitive values with redacted placeholders.
4. **Hallucination Detection:** Uses an LLM-based evaluator to flag unsupported responses.
5. **Toxicity Detection:** Evaluates generated responses for harmful or abusive content.

## 🔄 RAG Workflow

1. Upload multiple PDF documents.
2. Extract document text using PyPDFLoader.
3. Split documents into overlapping chunks.
4. Generate embeddings using Gemini Embedding.
5. Store embeddings in a FAISS vector index.
6. Retrieve the top 4 relevant chunks for each query.
7. Generate context-grounded answers using Gemini Flash.
8. Apply output guardrails and sensitive information masking.
9. Evaluate retrieval and response quality.
10. Display answers, sources, evaluation metrics, and performance statistics.

## 🎯 Project Objectives

- Build an end-to-end multi-document RAG application.
- Improve answer reliability through context-grounded generation.
- Implement security guardrails for safer LLM interactions.
- Evaluate retrieval and generation quality.
- Monitor LLM performance and token consumption.

## 🔮 Future Enhancements

- Support additional document formats such as DOCX and TXT.
- Implement hybrid search combining semantic and keyword retrieval.
- Add reranking for improved retrieval relevance.
- Introduce advanced automated evaluation frameworks.
- Add persistent vector database storage.
- Implement user authentication and chat history management.

## 👨‍💻 Author

**Anuj Verma**

Aspiring Data Scientist | Generative AI & Agentic AI Enthusiast

**Core Interests:** Machine Learning, Deep Learning, Generative AI, RAG Systems, and Agentic AI.
