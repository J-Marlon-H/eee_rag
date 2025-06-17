# Reeeliance Internal RAG Chatbot

![Chatbot Demo](reeeliance-logo-rgb@1x.png) This repository contains the source code for the Reeeliance internal chatbot. This tool leverages a Retrieval-Augmented Generation (RAG) architecture to answer questions based on a private knowledge base of company documents.

The application is built with Python using **Streamlit** for the user interface and **LangChain** to orchestrate the AI components.

## Table of Contents
- [High-Level Overview](#high-level-overview)
- [How it Works: Key Concepts](#how-it-works-key-concepts)
  - [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
  - [Vector Stores (Pinecone)](#vector-stores-pinecone)
  - [Embeddings (Azure OpenAI)](#embeddings-azure-openai)
  - [LLMs (Azure AI)](#llms-azure-ai)
  - [LangChain Chains](#langchain-chains)
- [Technology Stack](#technology-stack)
- [Setup and Deployment](#setup-and-deployment)
  - [Prerequisites](#prerequisites)
  - [Configuration](#configuration)
  - [Running Locally](#running-locally)
- [Data Sources](#data-sources)

## High-Level Overview

The primary goal of this chatbot is to provide accurate, context-aware answers to questions about internal company processes and documentation. Instead of relying solely on a public Large Language Model (LLM), it first retrieves relevant information from our private documents and then uses the LLM to generate a helpful answer based on that specific information.

This approach ensures that answers are grounded in our company's data, reducing hallucinations and providing more relevant results.

The application is protected by a simple username/password login screen to ensure only authorized employees can access it.

## How it Works: Key Concepts

This application is built on several modern AI and application development concepts.

### Retrieval-Augmented Generation (RAG)

RAG is an advanced technique for building LLM-powered applications. Instead of just asking an LLM a question directly, the RAG process works like this:

1.  **Retrieve:** When a user asks a question, the system first searches through a private knowledge base (our internal documents) to find the most relevant snippets of text.
2.  **Augment:** These relevant text snippets (the "context") are then added to the user's original question.
3.  **Generate:** This combined package of context and question is sent to an LLM, with instructions to generate an answer based *on the provided context*.

### Vector Stores (Pinecone)

To enable fast and effective searching, our documents are not stored as plain text. They are converted into numerical representations called **embeddings** and stored in a specialized database called a **Vector Store**.

- **Pinecone** is the cloud-based vector store we use. It is highly optimized for finding the most similar vectors to a given query vector, which is how we find relevant document chunks.

### Embeddings (Azure OpenAI)

An "embedding" is a vector (a list of numbers) that captures the semantic meaning of a piece of text. Texts with similar meanings will have similar vectors.

- We use **Azure OpenAI's `text-embedding-3-small` model** to convert both our source documents and the user's questions into these numerical vectors for comparison.

### LLMs (Azure AI)

The Large Language Model (LLM) is the "brain" that generates the final, human-readable answer.

- We use **Azure AI** to host powerful models like **GPT-4o**. The LLM receives the prompt (which includes the user's question and the retrieved context) and synthesizes the final response.

### LangChain Chains

**LangChain** is a framework that helps us connect all these components together. A "chain" in LangChain is a sequence of steps.

- **`ConversationalRetrievalChain`**: This is the specific chain we use. It orchestrates the entire RAG workflow: it takes the user's question, uses a **Retriever** to fetch documents from the vector store, stuffs them into a **Prompt Template**, sends it to the **LLM**, and returns the final answer. It also has memory to handle follow-up questions.
- **`ContextualCompressionRetriever`**: Before feeding documents to the LLM, we use this retriever with **Cohere Rerank** to take the initial search results from Pinecone and re-rank them for the highest relevance, ensuring the LLM gets only the best possible context.

## Technology Stack

- **Application Framework:** [Streamlit](https://streamlit.io/)
- **AI Orchestration:** [LangChain](https://www.langchain.com/)
- **LLM Provider:** [Azure AI Studio](https://azure.microsoft.com/en-us/products/ai-studio)
- **Embedding Model:** Azure OpenAI
- **Vector Store:** [Pinecone](https://www.pinecone.io/)
- **Reranking:** [Cohere](https://cohere.com/)

## Setup and Deployment

### Prerequisites

- Python 3.9+
- A GitHub account
- A Streamlit Community Cloud account

### Configuration

All API keys and credentials should be stored as secrets. For local development, create a file at `.streamlit/secrets.toml`. For deployment, add these secrets to the Streamlit Community Cloud dashboard.

**Required Secrets:**
```toml
# .streamlit/secrets.toml

# App Login
APP_USER = "your_app_username"
APP_PASSWORD = "your_app_password"

# Pinecone
PINECONE_API_KEY = "..."
PINECONE_INDEX_NAME = "..."

# Cohere
COHERE_API_KEY = "..."

# Azure
AZURE_INFERENCE_ENDPOINT = "..."
AZURE_OPENAI_ENDPOINT = "..."
AZURE_INFERENCE_CREDENTIAL = "..." 
# Note: AZURE_OPENAI_API_KEY is set from this credential in the app
```

### Running Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/J-Marlon-H/eee_rag.git](https://github.com/J-Marlon-H/eee_rag.git)
    cd eee_rag
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On macOS/Linux
    myenv\Scripts\activate    # On Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Create your local `.streamlit/secrets.toml` file** with the necessary keys.

5.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

## Data Sources

The knowledge base for this chatbot is currently populated with information from the following sources:

-   The project guidelines (from Teamworks)
-   The Quote template (from Google Drive)

This list will be expanded over time to include more internal documentation.