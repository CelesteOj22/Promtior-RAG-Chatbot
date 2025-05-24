import os
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langserve import add_routes

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Cargar variables de entorno\load_dotenv = load_dotenv
load_dotenv()

# Configuración desde .env
urls = os.getenv("SCRAPE_URLS", "").split(",")
pdf_path = os.getenv("PDF_PATH", "AI Engineer.pdf")
index_path = os.getenv("VECTOR_INDEX_PATH", "promtior_index")
model_name = os.getenv("OLLAMA_MODEL", "llama2")

# Plantillas y FastAPI\BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app = FastAPI()

# Función para cargar documentos
def load_documents():
    docs = []
    for url in urls:
        url = url.strip()
        if url:
            docs.extend(WebBaseLoader(url).load())
    if Path(pdf_path).exists():
        docs.extend(PyPDFLoader(pdf_path).load())
    return docs

# Preparar o cargar índice FAISS
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if not Path(index_path).exists():
        print("Índice no encontrado, creando uno nuevo...")
        documents = load_documents()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(index_path)
    else:
        print("Índice encontrado, cargando...")
        vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vs

# Instanciar vectorstore, LLM y RAG chain
vectorstore = get_vectorstore()
llm = OllamaLLM(model=model_name)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Agregar ruta RAG
add_routes(app, rag_chain, path="/chat")

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "answer": None})

@app.post("/", response_class=HTMLResponse)
def handle_form(request: Request, question: str = Form(...)):
    # Ejecutar la cadena y obtener la respuesta
    output = rag_chain.invoke({"query": question})
    answer = output.get("result")
    return templates.TemplateResponse(
        "form.html",
        {"request": request, "answer": answer, "question": question}
    )
