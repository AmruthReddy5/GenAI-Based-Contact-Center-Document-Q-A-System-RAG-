# GenAI-Based Contact Center Document Q&A System (RAG)

An AI-powered Retrieval-Augmented Generation (RAG) application that allows a contact center to automatically answer customer questions using company documentation.

Instead of manually searching policy documents, the system retrieves the most relevant sections and generates accurate answers using a Large Language Model.


# PROJECT OVERVIEW

Customer support teams rely on large documentation such as:

- Refund policies
- Shipping FAQs
- Account management guides

Searching these documents manually is time-consuming.

This project builds an AI-powered Q&A system that:

1. Converts company documents into vector embeddings
2. Stores them in a FAISS vector index
3. Retrieves the most relevant document sections
4. Uses an LLM to generate accurate answers based on those documents

This approach is called Retrieval-Augmented Generation (RAG).


# SYSTEM ARCHITECTURE

User Question
↓
Convert Question → Embedding
↓
FAISS Vector Search (Top-K chunks)
↓
Build Prompt (Context + Question)
↓
LLM Generates Answer
↓
Return Answer + Source Documents


# TECH STACK

Python  
Core programming language used to build the entire application.

OpenAI API  
Used for generating embeddings and natural language answers.

FAISS  
Facebook AI Similarity Search library used for fast vector search.

FastAPI  
Used to expose the system as a REST API.

MySQL  
Stores document metadata and query logs.

NumPy  
Used for vector and numerical operations.

Pandas  
Used for displaying query logs and results.

Uvicorn  
ASGI server used to run the FastAPI application.


# KEY FEATURES

- Semantic search using embeddings
- Retrieval-Augmented Generation pipeline
- Fast vector search using FAISS
- REST API using FastAPI
- Query logging using MySQL
- Document metadata tracking
- Swagger API documentation
- Category filtered search


# PROJECT STRUCTURE

RAG-ContactCenter-System

RAG_ContactCenter_Code.ipynb  
Main notebook containing the full implementation.

RAG_ContactCenter_Demo.ipynb  
Notebook used for testing the system.

RAG_Project_Explanation.pdf  
Detailed explanation of the project architecture and workflow.

faiss_index.bin  
Saved FAISS vector index.

chunks_metadata.json  
Maps FAISS vector index positions to document chunks.

main.py  
FastAPI application that exposes the RAG system as REST APIs.


# RAG PIPELINE

The system follows the standard Retrieve → Augment → Generate pipeline.

STEP 1 – RETRIEVE

The user question is converted into a vector embedding and FAISS searches for the most similar document chunks.

STEP 2 – AUGMENT

The retrieved chunks are inserted into the prompt together with the user question.

STEP 3 – GENERATE

The language model generates an answer using the provided context only.

This prevents hallucination and ensures the response is grounded in company documents.


# EXAMPLE QUERY

Question

How long does a refund take?

Retrieved Source

Refund and Return Policy

Generated Answer

Our company offers a 30-day return policy. Once the returned item is received and inspected, refunds are processed within 5–7 business days.


# INSTALLATION

Install dependencies

pip install openai faiss-cpu mysql-connector-python fastapi uvicorn  
pip install nltk numpy pandas tiktoken python-dotenv requests


# RUN THE NOTEBOOK

Open the notebook using:

VS Code with Jupyter extension  
or  
Jupyter Notebook / Jupyter Lab

Run all cells sequentially.


# RUN THE API SERVER

uvicorn main:app --reload --port 8000


# OPEN API DOCUMENTATION

Open the following URL in your browser

http://localhost:8000/docs

Swagger UI will appear where you can test all endpoints interactively.


# API ENDPOINTS

GET /

Health check endpoint that returns server status and model information.

POST /query

Submit a question and receive an AI-generated answer with document sources.

GET /documents

Returns a list of all ingested documents.

GET /analytics

Returns query statistics such as response times and recent queries.


# EXAMPLE API REQUEST

POST /query

Request

{
"question": "What is the refund policy?"
}

Response

{
"answer": "...",
"sources": [...],
"metadata": {...}
}


SAMPLE DOCUMENTS USED
Refund and Return Policy  
Shipping and Delivery FAQ  
Account Management Guide

These documents are cleaned, chunked, and converted into embeddings before indexing.


# PERFORMANCE
Typical response time: 700 ms – 1500 ms
depending on model and query complexity.

# FUTURE IMPROVEMENTS
Add a frontend chat interface  
Support PDF and Word document ingestion  
Use a vector database such as Pinecone or Weaviate  
Implement authentication and access control  
Deploy the system to cloud infrastructure

