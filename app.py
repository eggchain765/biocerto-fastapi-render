import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# ✅ Istanzia FastAPI solo UNA volta
app = FastAPI(title="Biocerto.AI - RAG")

# ✅ Abilita il CORS (permette richieste dal frontend su biocerto.it)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # puoi sostituire "*" con ["https://biocerto.it"] in produzione
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📄 Caricamento dei documenti PDF dalla cartella data/
documents = []
data_path = "data"
os.makedirs(data_path, exist_ok=True)
for file in os.listdir(data_path):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(data_path, file))
        documents.extend(loader.load())

# ✂️ Divisione in chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

# 🔎 Embedding + indicizzazione FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedding_model)

# 🧠 LLM Pipeline
hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 🔗 Catena RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# 📩 Modello input
class Question(BaseModel):
    query: str

# ✅ Endpoint API
@app.post("/ask")
def ask_question(payload: Question):
    response = qa_chain.run(payload.query)
    return {"answer": response}
