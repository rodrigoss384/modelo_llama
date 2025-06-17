import os
import time
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Constantes
KNOWLEDGE_BASE_DIR = "base_conhecimento"
FAISS_INDEX_PATH = "faiss_index"

def create_vector_store():
    """
    Lê documentos de diferentes formatos de um diretório, os processa 
    e cria um índice FAISS para busca de similaridade.
    """
    print("Iniciando a criação da base de conhecimento...")

    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"ERRO: A pasta '{KNOWLEDGE_BASE_DIR}' não foi encontrada.")
        print("Por favor, crie esta pasta e coloque seus documentos nela.")
        return

    all_documents = []
    start_time = time.time()
    
    # CORREÇÃO: Itera sobre os arquivos no diretório para usar o loader correto.
    print(f"Buscando arquivos em '{KNOWLEDGE_BASE_DIR}'...")
    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        
        try:
            if filename.endswith(".pdf"):
                print(f"  - Carregando PDF: {filename}")
                loader = PyPDFLoader(file_path)
                all_documents.extend(loader.load())
            elif filename.endswith(".txt"):
                print(f"  - Carregando TXT: {filename}")
                loader = TextLoader(file_path, encoding='utf-8')
                all_documents.extend(loader.load())
            # Adicione outras condições aqui para mais tipos de arquivo (ex: .docx, .csv)
            # elif filename.endswith(".docx"):
            #     # from langchain_community.document_loaders import Docx2txtLoader
            #     loader = Docx2txtLoader(file_path)
            #     all_documents.extend(loader.load())
        except Exception as e:
            print(f"    ERRO ao carregar o arquivo {filename}: {e}")

    if not all_documents:
        print("Nenhum documento foi carregado. Verifique os arquivos na pasta 'base_conhecimento'. Encerrando.")
        return
        
    end_time = time.time()
    print(f"\nDocumentos carregados em {end_time - start_time:.2f} segundos. Total de {len(all_documents)} páginas/documentos.")

    # Divide os documentos em pedaços menores (chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    print("Dividindo documentos em chunks...")
    docs = text_splitter.split_documents(all_documents)
    print(f"Total de {len(docs)} chunks criados.")

    # Define o modelo de embeddings
    print("Carregando modelo de embeddings (pode baixar na primeira vez)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    print("Modelo de embeddings carregado.")

    # Cria o índice FAISS
    print("Criando o índice FAISS... Isso pode levar alguns minutos dependendo do volume de documentos.")
    start_time = time.time()
    db = FAISS.from_documents(docs, embeddings)
    end_time = time.time()
    print(f"Índice criado em {end_time - start_time:.2f} segundos.")

    # Salva o índice localmente
    db.save_local(FAISS_INDEX_PATH)
    print(f"Base de conhecimento salva com sucesso em '{FAISS_INDEX_PATH}'!")

if __name__ == "__main__":
    create_vector_store()

