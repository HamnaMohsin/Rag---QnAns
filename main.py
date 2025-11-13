# main.py
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# 1️⃣ Load speech text
loader = TextLoader("speech.txt", encoding="utf-8")
documents = loader.load()

# 2️⃣ Split text into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=500,  # adjust chunk size if needed
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# 3️⃣ Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4️⃣ Create or load local Chroma vector store
vectordb = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")
vectordb.persist()

# 5️⃣ Setup Ollama LLM
llm = Ollama(model="mistral")

# 6️⃣ Create Retrieval QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",   # simple approach
    retriever=vectordb.as_retriever()
)

# 7️⃣ Interactive Q&A loop
print("💡 Welcome to Dr. B.R. Ambedkar Speech Q&A. Type 'exit' to quit.")

while True:
    query = input("\nEnter your question: ")
    if query.lower() in ["exit", "quit"]:
        break
    answer = qa.run(query)
    print("\n📝 Answer:", answer)
