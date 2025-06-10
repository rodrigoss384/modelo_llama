import json
import uuid
from datetime import datetime
import os
import pandas as pd
import PyPDF2
from llama_cpp import Llama

# Caminho para o modelo GGUF (ajuste conforme necessário)
MODEL_PATH = "llama-2-7b-chat-Q6.gguf" # Mantive o nome como você indicou no exemplo do prompt

# Carrega o modelo uma vez durante a inicialização
# Configurado para rodar SOMENTE na CPU (n_gpu_layers=0) e com contexto razoável
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,       # Aumentei o contexto para 4096, comum para 7B, mas ajuste se tiver problemas de RAM.
    n_threads=os.cpu_count() or 6, # Usa todos os cores da CPU disponíveis para melhor performance
    n_gpu_layers=0,   # ESSENCIAL: 0 para rodar SOMENTE na CPU
    verbose=True      # Mantenha True durante o desenvolvimento para ver logs úteis
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
    # O contexto será adicionado como uma mensagem 'system' ou dentro da mensagem 'user'
    # Retornamos o texto puro aqui, o wrapping será feito em generate_chat_prompt
    return f"{source}:\n{context}"

def generate_chat_prompt(user_message, conversation_history=None, context=""):
    """
    Gera uma lista de mensagens formatadas para a API de chat do Llama 2 (OpenAI format).
    """
    system_prompt_content = """Você é um assistente especializado em Suporte de TI e Segurança da Informação, com foco profundo em **redes de computadores (incluindo switches, roteadores, Wi-Fi, cabeamento)** e **segurança de rede (especificamente firewalls, VPNs, IPS/IDS)**.

Sua principal função é:
1.  **Diagnosticar e solucionar problemas:** Oferecer passos detalhados para identificar e resolver falhas de conectividade, configuração e desempenho em equipamentos de rede e firewalls.
2.  **Fornecer recomendações de segurança:** Aconselhar sobre melhores práticas, configurações seguras e mitigação de ameaças para firewalls e infraestruturas de rede.
3.  **Explicar conceitos técnicos:** Apresentar informações claras e concisas sobre tecnologias de rede e segurança, adaptadas ao nível de conhecimento do usuário (se possível).
4.  **Ser preciso e factual:** Baseie suas respostas em conhecimento técnico estabelecido e, se não souber, diga que não tem a informação.
5.  **Priorizar a segurança:** Sempre que uma sugestão puder impactar a segurança, ressalte a importância de testes em ambiente controlado ou consulta a um especialista.

**Evite:**
* Aconselhar sobre problemas que exigem acesso físico sem supervisão (ex: manipular hardware diretamente sem orientação).
* Gerar comandos específicos de configuração para ambientes desconhecidos sem alertar sobre riscos.
* Discutir tópicos fora de sua especialidade (ex: programação de software, desenvolvimento web, marketing digital).
* Alucinações ou informações falsas.
* Linguagem informal ou excessivamente técnica sem necessidade.

IMPORTANTE: Você DEVE sempre tentar ajudar tecnicamente, nunca redirecionar para suporte externo.
"""
    
    messages = [
        {"role": "system", "content": system_prompt_content}
    ]

    # Adicionar histórico de conversa.
    # Assumimos que 'conversation_history' já está no formato [{'role': 'user/assistant', 'content': '...'}]
    # ou que o app.py irá formatá-lo assim.
    # A biblioteca llama_cpp_python cuidará dos tokens [INST] e [/INST]
    if conversation_history:
        # Pega as últimas 6 mensagens e as formata para o Llama 2
        # É importante que o histórico tenha o 'role' correto ('user' ou 'assistant')
        # E que o LLM entenda o formato de conversação
        # A API create_chat_completion do llama-cpp-python já faz a formatação Llama 2
        # se o modelo for um chat model.
        # Então, basta adicionar as mensagens como elas são.
        messages.extend(conversation_history)

    # Adicionar contexto se fornecido, antes da mensagem atual do usuário.
    # Se o contexto é para ser parte do prompt "principal", adiciona-o como uma mensagem de sistema
    # ou como parte da mensagem do usuário. Para RAG, geralmente se adiciona como parte da instrução do usuário.
    if context and context.strip():
        # Adicionar o contexto diretamente à mensagem do usuário para garantir que o modelo o utilize.
        # Ou, se o contexto for muito grande, considere passá-lo como uma mensagem de sistema adicional
        # ou como parte da primeira mensagem do usuário.
        # Para RAG, a forma mais comum é injetar no turno atual.
        user_message_with_context = f"{context}\n\n{user_message}"
        messages.append({"role": "user", "content": user_message_with_context})
    else:
        messages.append({"role": "user", "content": user_message})
    
    return messages

def invoke_local_model(messages, model_params=None):
    """
    Invoca o modelo Llama local com a lista de mensagens fornecida, usando create_chat_completion.
    """
    if model_params is None:
        model_params = {
            "temperature": 0.3,
            "top_p": 0.7,
            "top_k": 30,
            "max_tokens": 900
        }

    try:
        if not isinstance(messages, list) or not messages:
            raise ValueError("Mensagens inválidas ou vazias. Espera-se uma lista de dicionários.")
              
        # Usar create_chat_completion, que é a forma recomendada para modelos de chat como o Llama 2
        # Ele cuida da formatação de tokens (<s>, [INST], etc.) automaticamente.
        response = llm.create_chat_completion(
            messages=messages,
            temperature=model_params["temperature"],
            max_tokens=model_params["max_tokens"],
            top_p=model_params["top_p"],
            top_k=model_params["top_k"],
            # 'stop' pode ser ajustado aqui, mas o modelo de chat já é bom em parar
            # Evite stops como "\nUsuário:" se você quer que o modelo continue a resposta sem ser interrompido
            # por uma linha que ele poderia gerar.
            stop=["\nUsuário:", "###", "</s>"], # Adicione </s> para garantir que ele pare no fim da sequência
            # echo=False e repeat_penalty são passados para o método de chat_completion
            # Estes são parâmetros válidos para o create_chat_completion
            # repeat_penalty=1.1 # Descomente se achar que as respostas são repetitivas
        )
        
        # Verificar se a resposta é válida
        if not response or 'choices' not in response or not response['choices']:
            raise ValueError("Resposta inválida do modelo")
            
        # A resposta de create_chat_completion é diferente da inferência de texto bruto
        answer = response['choices'][0]['message']['content'].strip()
        
        if not answer:
            answer = "Para ajudá-lo melhor, preciso de mais informações. Pode descrever o problema em detalhes?"
        
        return {
            "answer": answer,
            "sessionId": str(uuid.uuid4()) # Gera um novo UUID para a sessão se necessário
        }
        
    except Exception as e:
        print(f"ERRO DETALHADO na invocação do modelo: {str(e)}")
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
