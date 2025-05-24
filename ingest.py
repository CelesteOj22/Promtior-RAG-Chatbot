import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Cargar variables de entorno
load_dotenv()

# Leer variables
urls = os.getenv("DOCUMENT_URLS", "")
pdf_path = os.getenv("PDF_FILE", "")
index_path = os.getenv("VECTOR_INDEX_PATH", "vector_index")

# Inicializar lista de documentos
all_docs = []

# Cargar desde URLs
url_list = [url.strip() for url in urls.split(",") if url.strip()]
for url in url_list:
    print(f"Cargando contenido de: {url}")
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        all_docs.extend(docs)
    except Exception as e:
        print(f"Error al cargar {url}: {e}")

# Cargar desde PDF
if pdf_path and os.path.exists(pdf_path):
    print(f"Cargando contenido del PDF: {pdf_path}")
    try:
        pdf_loader = PyPDFLoader(pdf_path)
        pdf_docs = pdf_loader.load()
        all_docs.extend(pdf_docs)
    except Exception as e:
        print(f"Error al cargar PDF '{pdf_path}': {e}")
else:
    print("No se encontró el PDF o no se especificó.")

# Dividir documentos en fragmentos
print("Dividiendo documentos...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

# Crear embeddings y FAISS index
print("Generando embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Guardar índice
vectorstore.save_local(index_path)
print(f"Índice guardado en: {index_path}")
