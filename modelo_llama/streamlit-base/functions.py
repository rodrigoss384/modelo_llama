import json
import uuid
from datetime import datetime
import os
import pandas as pd
import PyPDF2
from llama_cpp import Llama

MODEL_PATH = "llama-2-7b-chat.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=os.cpu_count() or 6,
    n_gpu_layers=0,
    verbose=True
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
    return f"{source}:\n{context}"

def generate_chat_prompt(user_message, conversation_history=None, context=""):
    """
    Gera uma lista de mensagens formatadas para a API de chat do Llama 2 (OpenAI format).
    """
    # ALTERAÇÃO: O prompt do sistema foi reescrito para ser mais direto e claro,
    # incluindo uma regra explícita e crucial sobre o idioma da resposta.
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
    Invoca o modelo Llama local com a lista de mensagens fornecida, usando create_chat_completion.
    """
    if model_params is None:
        # ALTERAÇÃO: Unifiquei e ajustei os parâmetros para focar em precisão e performance.
        model_params = {
            "temperature": 0.2, # Mais baixo para respostas mais factuais
            "top_p": 0.8,
            "top_k": 20,          # Reduzido para focar nas opções mais prováveis
            "max_tokens": 800     # Reduzido para evitar respostas excessivamente longas e lentas
        }

    try:
        if not isinstance(messages, list) or not messages:
            raise ValueError("Mensagens inválidas ou vazias. Espera-se uma lista de dicionários.")
              
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
        
        if not answer:
            answer = "Não consegui gerar uma resposta. Poderia reformular sua pergunta ou fornecer mais detalhes?"
        
        return {
            "answer": answer,
            "sessionId": str(uuid.uuid4())
        }
        
    except Exception as e:
        print(f"ERRO DETALHADO na invocação do modelo: {str(e)}")
        return {
            "error": str(e),
            "answer": f"Ocorreu um erro crítico ao se comunicar com o modelo: {str(e)}",
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