import json
import uuid
from datetime import datetime
import os
import pandas as pd
import PyPDF2
from llama_cpp import Llama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Carrega o modelo uma vez durante a inicialização
# ATENÇÃO: Certifique-se que o caminho para o seu modelo GGUF está correto.
MODEL_PATH = "llama-2-7b-chat.gguf"

# Um bloco try-except aqui pode ajudar a dar uma mensagem de erro mais clara se o modelo não for encontrado.
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=os.cpu_count() or 6,
        n_gpu_layers=0,
        verbose=True
    )
except ValueError as e:
    print("="*50)
    print(f"ERRO CRÍTICO: Não foi possível carregar o modelo em '{MODEL_PATH}'.")
    print("Verifique se o caminho está correto e o arquivo não está corrompido.")
    print(f"Detalhe do erro: {e}")
    print("="*50)
    # Encerra o script se o modelo não puder ser carregado.
    exit()


FAISS_INDEX_PATH = "faiss_index"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Carrega o índice FAISS se ele existir
if os.path.exists(FAISS_INDEX_PATH):
    print("Carregando base de conhecimento (FAISS)...")
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Base de conhecimento carregada com sucesso.")
else:
    db = None
    print("AVISO: Base de conhecimento 'faiss_index' não encontrada. A função de busca estará desativada.")

def search_knowledge_base(query: str, k: int = 4) -> str:
    """
    Busca na base de conhecimento FAISS os chunks mais relevantes para a query.
    """
    if db is None:
        return "A base de conhecimento não está disponível."
    
    print(f"Buscando por: '{query}' na base de conhecimento...")
    # Realiza a busca por similaridade
    results = db.similarity_search(query, k=k)
    
    # Formata os resultados para incluir no prompt
    context = "\n\n---\n\n".join([doc.page_content for doc in results])
    return context

def read_pdf_from_uploaded_file(uploaded_file):
    """Lê o conteúdo de um arquivo PDF carregado pelo Streamlit."""
    try:
        import io
        from PyPDF2 import PdfReader
        
        pdf_bytes = io.BytesIO(uploaded_file.getvalue())
        reader = PdfReader(pdf_bytes)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Erro ao ler PDF: {str(e)}"

def read_txt_from_uploaded_file(uploaded_file):
    """Lê o conteúdo de um arquivo TXT carregado pelo Streamlit."""
    try:
        return uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        return f"Erro ao ler TXT: {str(e)}"

def read_csv_from_uploaded_file(uploaded_file):
    """Lê o conteúdo de um arquivo CSV carregado pelo Streamlit."""
    try:
        import pandas as pd
        import io
        
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
        return df.to_string()
    except Exception as e:
        return f"Erro ao ler CSV: {str(e)}"

def format_context(context, source="Contexto Adicional"):
    """Formata o contexto para ser adicionado ao prompt."""
    return f"{source}:\n{context}"

def generate_chat_prompt(user_message, conversation_history=None, context=""):
    """
    Gera uma lista de mensagens formatadas para a API de chat do Llama 2 (OpenAI format).
    """
    system_prompt_content = """
Você é um assistente de TI ultra especializado em redes e segurança.

**SUAS REGRAS:**
1.  **IDIOMA:** Responda **EXCLUSIVAMENTE** em Português do Brasil. Nunca, sob nenhuma circunstância, use inglês ou qualquer outro idioma.
2.  **ESPECIALIDADE:** Seu foco total é em **redes de computadores** (switches, roteadores, Wi-Fi, firewalls, VPNs, IPS/IDS) e **segurança da informação** relacionada.
3.  **AÇÃO:** Seu objetivo é sempre diagnosticar problemas e fornecer soluções técnicas detalhadas. Seja direto e prático.
4.  **PRECISÃO:** Forneça informações factuais. Se não souber a resposta, admita que não possui a informação. Não invente.
5.  **PROIBIÇÕES:** Não discuta tópicos fora da sua especialidade (programação, marketing, etc.). Não redirecione o usuário para suporte externo; você é o suporte.

Comece a análise agora.
"""
    
    messages = [
        {"role": "system", "content": system_prompt_content}
    ]

    if conversation_history:
        messages.extend(conversation_history)

    if context and context.strip():
        user_message_with_context = f"**Use o seguinte contexto para formular sua resposta:**\n{context}\n\n**Pergunta do usuário:**\n{user_message}"
        messages.append({"role": "user", "content": user_message_with_context})
    else:
        messages.append({"role": "user", "content": user_message})
    
    return messages

def invoke_local_model(messages, model_params=None):
    """
    Invoca o modelo Llama local e retorna a resposta junto com a contagem de tokens.
    """
    if model_params is None:
        model_params = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": 800
        }

    try:
        if not isinstance(messages, list) or not messages:
            raise ValueError("Mensagens inválidas ou vazias.")
              
        response = llm.create_chat_completion(
            messages=messages,
            temperature=model_params["temperature"],
            max_tokens=model_params["max_tokens"],
            top_p=model_params["top_p"],
            top_k=model_params["top_k"],
            stop=["\nUsuário:", "###", "</s>"],
        )
        
        if not response or 'choices' not in response or not response['choices']:
            raise ValueError("Resposta inválida do modelo")
            
        answer = response['choices'][0]['message']['content'].strip()
        
        usage_data = response.get('usage', {})
        prompt_tokens = usage_data.get('prompt_tokens', 0)
        completion_tokens = usage_data.get('completion_tokens', 0)
        total_tokens = usage_data.get('total_tokens', 0)

        if not answer:
            answer = "Não consegui gerar uma resposta. Poderia reformular?"
        
        return {
            "answer": answer,
            "sessionId": str(uuid.uuid4()),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        
    except Exception as e:
        print(f"ERRO DETALHADO na invocação do modelo: {str(e)}")
        # CORREÇÃO: Garante que o dicionário retornado em caso de erro tenha a mesma estrutura.
        return {
            "error": str(e),
            "answer": f"Ocorreu um erro ao processar sua solicitação: {str(e)}",
            "sessionId": str(uuid.uuid4()),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

def get_model_context_size():
    """Retorna o tamanho da janela de contexto (n_ctx) do modelo carregado."""
    return llm.n_ctx()
