# Rag---QnAns
# Dr. B.R. Ambedkar Speech Q&A System

A **command-line Question & Answer system** based on Dr. B.R. Ambedkar's speech excerpt from *"Annihilation of Caste"*. This system uses a **Retrieval-Augmented Generation (RAG)** pipeline to answer questions using the speech text, powered by **LangChain**, **ChromaDB**, **HuggingFace embeddings**, and **Ollama Mistral 7B**.  

The system is **fully local**, **free**, and requires **no API keys**.

---

## Features

- Loads the speech text from a local file (`speech.txt`)  
- Splits text into manageable chunks for embeddings  
- Generates vector embeddings using **sentence-transformers/all-MiniLM-L6-v2**  
- Stores embeddings in **ChromaDB** for retrieval  
- Answers questions using **Ollama Mistral 7B**  
- Interactive command-line interface for Q&A  

---

## Project Structure

