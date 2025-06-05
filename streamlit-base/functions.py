import json
import uuid
from datetime import datetime
import os
import pandas as pd
import PyPDF2
from llama_cpp import Llama

# Caminho para o modelo GGUF (ajuste conforme necessário)
MODEL_PATH = "llama-2-7b.Q4_0.gguf"

# Carrega o modelo uma vez durante a inicialização
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,       # Contexto máximo
    n_threads=4,      # Número de threads
    n_gpu_layers=0    # Camadas para GPU (0 para CPU)
)

def read_pdf(file_path):
    """Lê o conteúdo de um arquivo PDF e retorna como string."""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Erro ao ler PDF: {str(e)}"

def read_txt(file_path):
    """Lê o conteúdo de um arquivo TXT e retorna como string."""
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Erro ao ler TXT: {str(e)}"

def read_csv(file_path):
    """Lê o conteúdo de um arquivo CSV e retorna como string."""
    try:
        df = pd.read_csv(file_path)
        return df.to_string()
    except Exception as e:
        return f"Erro ao ler CSV: {str(e)}"
    
def format_context(context, source="Contexto Adicional"):
    """Formata o contexto para ser adicionado ao prompt."""
    return f"\n\n{source}:\n{context}\n\n"

def generate_chat_prompt(user_message, conversation_history=None, context=""):
    """
    Gera um prompt de chat completo com histórico de conversa e contexto opcional.
    """
    system_prompt = """
    Você é um Assistente virtual
    Ajuda usuários na resolução de problemas e retirada de dúvidas
    """

    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        conversation_context = "Histórico da conversa:\n"
        recent_messages = conversation_history[-8:]
        for message in recent_messages:
            role = "Usuário" if message.get('role') == 'user' else "Assistente"
            conversation_context += f"{role}: {message.get('content')}\n"
        conversation_context += "\n"

    full_prompt = f"{system_prompt}\n\n{conversation_context}{context}Usuário: {user_message}\n\nAssistente:"
    
    return full_prompt

def invoke_local_model(prompt, model_params=None):
    """
    Invoca o modelo Llama local com o prompt fornecido.
    """
    if model_params is None:
        model_params = {
            "temperature": 0.7,
            "top_p": 0.85,
            "top_k": 40,
            "max_tokens": 800
        }

    try:
        response = llm(
            prompt,
            max_tokens=model_params["max_tokens"],
            temperature=model_params["temperature"],
            top_p=model_params["top_p"],
            top_k=model_params["top_k"],
            stop=["\n", "###"],
            echo=False
        )
        
        answer = response['choices'][0]['text'].strip()
        
        return {
            "answer": answer,
            "sessionId": str(uuid.uuid4())
        }
        
    except Exception as e:
        print(f"ERRO: Falha na invocação do modelo local: {str(e)}")
        return {
            "error": str(e),
            "answer": f"Ocorreu um erro ao processar sua solicitação: {str(e)}",
            "sessionId": str(uuid.uuid4())
        }

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