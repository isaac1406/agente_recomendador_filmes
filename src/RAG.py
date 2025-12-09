import os
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 

def build_vector_store(csv_path, api_key_ignorada):
    """Gerencia banco vetorial local usando embeddings HuggingFace."""
    pasta_banco = "faiss_index_filmes"
    
    # Inicializa modelo de embeddings local
    print("🔄 Init embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 1. Carrega índice existente do disco
    if os.path.exists(pasta_banco):
        print("🚀 Carregando banco local...")
        vectorstore = FAISS.load_local(
            pasta_banco, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vectorstore.as_retriever(search_kwargs={"k": 5})

    # 2. Cria novo índice se não existir
    print("⚠️ Criando novo índice...")
    
    # Carrega dados do CSV
    loader = CSVLoader(
        file_path=csv_path, 
        source_column="title", 
        encoding="utf-8"
    )
    docs = loader.load()
    
    # Gera vetores e salva no disco
    print(f"🧠 Processando {len(docs)} itens...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(pasta_banco)
    
    print("💾 Banco salvo.")
    return vectorstore.as_retriever(search_kwargs={"k": 5})