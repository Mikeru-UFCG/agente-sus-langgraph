import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_documents():
    print("1. Carregando PDFs da pasta 'docs'...")
    # Lê todos os PDFs da pasta docs
    loader = PyPDFDirectoryLoader("docs")
    docs = loader.load()
    
    if not docs:
        print("Erro: Nenhum PDF encontrado na pasta 'docs'!")
        return

    print(f"Encontradas {len(docs)} páginas. Quebrando em chunks...")
    # Quebra o texto em pedaços de 500 caracteres (ideal para LLMs menores)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Gerados {len(chunks)} chunks de texto.")

    print("2. Baixando/Carregando modelo de Embeddings (HuggingFace)...")
    # Usa o modelo bge-small sugerido pelo professor (roda bem em CPU)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("3. Criando o banco de dados vetorial FAISS...")
    # Gera os vetores e salva no FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Salva o banco localmente para não ter que processar toda vez
    vectorstore.save_local("faiss_index")
    print("Sucesso! Banco FAISS salvo na pasta 'faiss_index'.")

if __name__ == "__main__":
    ingest_documents()