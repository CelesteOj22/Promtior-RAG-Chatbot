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
from langchain.chains import RetrievalQA
from langchain_together import Together

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Cargar variables de entorno\load_dotenv = load_dotenv
load_dotenv()

# Configuración desde .env
urls = os.getenv("SCRAPE_URLS", "").split(",")
pdf_path = BASE_DIR / os.getenv("PDF_PATH", "AI Engineer.pdf")
index_path = os.getenv("VECTOR_INDEX_PATH", "promtior_index")

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
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        # Solo incluir la página 3 About Us (índice 20)
        docs.append(pages[2])
    return docs

# Preparar o cargar índice FAISS
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    """
    # ⚠️ Eliminar el índice existente para forzar la regeneración
    if Path(index_path).exists():
        shutil.rmtree(index_path)
        print("Índice anterior eliminado. Regenerando...")

        # Cargar documentos y dividirlos
        documents = load_documents()
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # Crear índice
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(index_path)
    """

    if not Path(index_path).exists():
        print("Índice no encontrado, creando uno nuevo...")


        # Cargar documentos y dividirlos
        documents = load_documents()
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # Crear índice
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(index_path)
    else:
        print("Índice encontrado, cargando...")
        vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vs

# Instanciar vectorstore, LLM y RAG chain
vectorstore = get_vectorstore()
llm = Together(
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.7,
    max_tokens=512,
    together_api_key=os.getenv("TOGETHER_API_KEY")
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)
"""
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
    return_source_documents=True,
    chain_type="refine"
)"""
# Agregar ruta RAG
add_routes(app, rag_chain, path="/chat")

chat_history = []

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "history": chat_history})

@app.post("/", response_class=HTMLResponse)
def handle_form(request: Request, question: str = Form(...)):
    output = rag_chain.invoke({"query": question})
    answer = output.get("result")

    # Agregar a historial
    chat_history.append({"question": question, "answer": answer})

    return templates.TemplateResponse(
        "form.html",
        {"request": request, "history": chat_history}
    )
