import streamlit as st
import uuid
import time
import hmac
import base64
from datetime import datetime
import re
import json
import os
from functions import (
    generate_chat_prompt, format_context, 
    read_pdf_from_uploaded_file, read_txt_from_uploaded_file, 
    read_csv_from_uploaded_file, invoke_local_model
)

def add_javascript():
    """Adiciona JavaScript para melhorar a interação do usuário com o chat"""
    js_code = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            const textarea = document.querySelector('textarea');
            if (textarea) {
                textarea.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        const sendButton = document.querySelector('button[data-testid="baseButton-secondary"]');
                        if (sendButton) {
                            sendButton.click();
                        }
                    }
                });
            }
        }, 1000);
    });
    </script>
    """
    st.components.v1.html(js_code, height=0)

st.set_page_config(
   page_title="Modelo de IA",
   page_icon="",
   layout="wide",
   initial_sidebar_state="expanded"
)

logo_path = "logo.png"

def preprocess_user_message(message):
    return message

def query_local_model(message, session_id="", model_params=None, context="", conversation_history=None):
    """
    Envia uma mensagem para o modelo local, usando a nova estrutura de mensagens.
    """
    if model_params is None:
        model_params = {
            "temperature": 0.3, # Ajustei a temperatura para um comportamento mais focado
            "top_p": 0.7,
            "top_k": 30,
            "max_tokens": 900
        }
    
    try:
        # generate_chat_prompt agora retorna uma LISTA de mensagens no formato {"role": "...", "content": "..."}
        messages_for_model = generate_chat_prompt(message, conversation_history=conversation_history, context=context)
        
        # invoke_local_model agora espera uma LISTA de mensagens
        result = invoke_local_model(messages_for_model, model_params)
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        result["sessionId"] = session_id
        return result
        
    except Exception as e:
        print(f"ERRO: Falha na requisição ao modelo local: {str(e)}")
        return {
            "error": str(e),
            "answer": "Ocorreu um erro ao processar sua solicitação.",
            "sessionId": session_id or str(uuid.uuid4())
        }

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        print(f"DEBUG LOGIN: Tentativa de login - Usuário: '{st.session_state['username']}', Senha: '{st.session_state['password']}'")
        
        if hmac.compare_digest(st.session_state["username"].strip(), "admin") and \
        hmac.compare_digest(st.session_state["password"].strip(), "admin123"):
            print("DEBUG LOGIN: Autenticação bem-sucedida")
            st.session_state["password_correct"] = True
            st.session_state["auth_cookie"] = {
                "user": "admin",
                "exp": time.time() + (7 * 24 * 60 * 60)
            }
            
            try:
                st.query_params["auth"] = base64.b64encode(json.dumps(st.session_state["auth_cookie"]).encode()).decode()
            except:
                pass
                
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            print(f"DEBUG LOGIN: Autenticação falhou - Usuário: '{st.session_state['username']}', Senha: '{st.session_state['password']}'")
            print(f"DEBUG LOGIN: Comparação - Usuário igual: {st.session_state['username'].strip() == 'admin'}")
            print(f"DEBUG LOGIN: Comparação - Senha igual: {st.session_state['password'].strip() == 'admin123'}")
            
            st.session_state["password_correct"] = False
            st.session_state["login_attempt"] = True

    if "auth_cookie" in st.session_state:
        if st.session_state["auth_cookie"].get("exp", 0) > time.time():
            st.session_state["password_correct"] = True
            return True
        else:
            del st.session_state["auth_cookie"]
    
    try:
        if "auth" in st.query_params:
            auth_data = json.loads(base64.b64decode(st.query_params["auth"]).decode())
            if auth_data.get("exp", 0) > time.time():
                st.session_state["auth_cookie"] = auth_data
                st.session_state["password_correct"] = True
                return True
    except:
        pass
    
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
        
    if "login_attempt" not in st.session_state:
        st.session_state["login_attempt"] = False

    if not st.session_state["password_correct"]:
        st.markdown("""
            <style>
                .stTextInput > div > div > input {
                    background-color: #f0f2f6;
                    color: #000000;
                }
                .login-form {
                    max-width: 400px;
                    margin: 0 auto;
                    padding: 2rem;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    background-color: white;
                }
                .login-title {
                    margin-bottom: 2rem;
                    text-align: center;
                    color: #4CAF50;
                }
                .login-button {
                    width: 100%;
                    margin-top: 1rem;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="login-form">', unsafe_allow_html=True)
        st.markdown('<h1 class="login-title">Login</h1>', unsafe_allow_html=True)
        
        st.text_input("Usuário", key="username")
        st.text_input("Senha", type="password", key="password")
        st.button("Entrar", on_click=password_entered, key="login-button")
        
        if st.session_state["login_attempt"] and not st.session_state["password_correct"]:
            st.error("Usuário ou senha incorretos")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return False
    else:
        return True

def logout():
    """Faz logout removendo o cookie de autenticação"""
    if "auth_cookie" in st.session_state:
        del st.session_state["auth_cookie"]
    st.session_state["password_correct"] = False
    st.session_state["login_attempt"] = False
    st.rerun()

def get_rag_context():
    """
    Obtém e formata o contexto RAG.    
    """
    context = ""
    if st.session_state.get('use_rag', False):
        if st.session_state.rag_source == "Arquivo":
            if st.session_state.uploaded_file is not None:
                file_type = st.session_state.file_type
                file = st.session_state.uploaded_file
                if file_type == "PDF":
                    return format_context(read_pdf_from_uploaded_file(file), f"Contexto do arquivo PDF: {file.name}")
                elif file_type == "TXT":
                    return format_context(read_txt_from_uploaded_file(file), f"Contexto do arquivo TXT: {file.name}")
                elif file_type == "CSV":
                    return format_context(read_csv_from_uploaded_file(file), f"Contexto do arquivo CSV: {file.name}")
        
        elif st.session_state.rag_source == "Texto Direto":
            if st.session_state.direct_text:
                return format_context(st.session_state.direct_text, "Contexto do Usuário")
    
    return context

# --- NOVA FUNÇÃO DE CALLBACK PARA O BOTÃO ENVIAR ---
def send_message_callback():
    """Lida com o envio da mensagem e limpa o campo de entrada."""
    if st.session_state.user_input and st.session_state.user_input.strip():
        # Captura o input atual
        temp_user_input = st.session_state.user_input
        
        # Limpa o input do text_area. Esta é a atribuição PERMITIDA.
        st.session_state.user_input = ""
        
        # Chama a lógica principal de processamento da mensagem
        _handle_message_logic(temp_user_input)
        
        # Resetar o uploader de arquivo se um arquivo foi enviado
        if st.session_state.get('file_to_send', None) is not None:
            st.session_state['file_to_send'] = None 
            st.session_state['file_uploader_key'] = str(uuid.uuid4()) # Gera nova chave para resetar o widget
        
        # Chame rerun para atualizar a UI após o processamento
        st.rerun()

# --- FUNÇÃO PRINCIPAL DE LÓGICA DE MENSAGEM (ANTIGA handle_message) ---
def _handle_message_logic(user_message_content):
    is_first_message = len(st.session_state.messages) == 0
    
    # Use user_message_content, não st.session_state.user_input
    user_message_raw = " ".join(user_message_content.strip().split())

    # Obter o contexto RAG
    rag_context = get_rag_context()
    
    # Adiciona o contexto técnico explicitamente (se desejar manter ele separado do RAG)
    tech_context = "CONEXÃO DE DOMÍNIO TÉCNICO:\n"
    tech_context += "- Redes e Firewalls\n- Segurança da Informação\n- Diagnóstico de Incidentes\n"
    
    # Combinando os contextos para passar para generate_chat_prompt.
    # generate_chat_prompt irá decidir como injetá-los no prompt final.
    combined_context_for_prompt = f"{rag_context}\n{tech_context}" if rag_context else tech_context

    # Verifica duplicidade ANTES de adicionar a mensagem ao histórico para evitar loop infinito
    is_duplicate = False
    if len(st.session_state.messages) > 0:
        last_user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
        if last_user_messages and last_user_messages[-1]["content"] == user_message_raw:
            is_duplicate = True
    
    if is_duplicate:
        # Se for duplicada, apenas saia. O input já foi limpo pelo callback.
        print(f"DEBUG: Mensagem duplicada detectada: '{user_message_raw}', ignorando.")
        return 

    # Lógica para anexos
    attached_file = st.session_state.get('file_to_send', None)
    file_content = ""
    file_info = ""
    user_message_display = user_message_raw # Mensagem para exibir no chat UI

    if attached_file is not None:
        file_extension = attached_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            file_content = read_pdf_from_uploaded_file(attached_file)
        elif file_extension == 'txt':
            file_content = read_txt_from_uploaded_file(attached_file)
        elif file_extension in ['csv', 'xls', 'xlsx']:
            file_content = read_csv_from_uploaded_file(attached_file)
        elif file_extension in ['doc', 'docx']:
            file_content = "Arquivo do Word anexado (processamento de conteúdo não disponível)"
        
        file_info = f"\n[Arquivo anexado: {attached_file.name}]"
        user_message_display = f"{user_message_raw}{file_info}" # Mensagem com anexo para exibição

    timestamp = datetime.now().strftime("%H:%M")
    
    # Adiciona a mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": user_message_display, "time": timestamp})
    
    # A sessão_id é passada para o query_local_model, não diretamente para generate_chat_prompt
    current_session_id = "" if is_first_message else st.session_state.session_id
    
    with st.chat_message("assistant", avatar=logo_path):
        typing_placeholder = st.empty()
        typing_placeholder.markdown("_Digitando..._")
        
        with st.spinner():
            # O contexto que será passado para generate_chat_prompt
            # Combinamos o contexto RAG e o conteúdo do arquivo, se houver.
            final_context_for_model = combined_context_for_prompt
            if file_content:
                file_context_formatted = format_context(file_content, f"Conteúdo do arquivo anexado ({attached_file.name})")
                final_context_for_model = f"{final_context_for_model}\n{file_context_formatted}"
            
            # Chamada para query_local_model que agora espera uma lista de mensagens
            # Passa o histórico da sessão ATUALIZADO (com a mensagem do usuário já adicionada)
            result = query_local_model(
                user_message_raw, # Passa a mensagem original do usuário (sem info de anexo para o modelo)
                current_session_id, 
                context=final_context_for_model, 
                conversation_history=st.session_state.messages # Passa o histórico completo para generate_chat_prompt
            )
        
        if result:
            assistant_message = result.get('answer', 'Não foi possível obter uma resposta.')
            
            if "sessionId" in result:
                new_session_id = result["sessionId"]
                print(f"DEBUG: API retornou sessionId: '{new_session_id}'")
                
                st.session_state.session_id = new_session_id
                print(f"DEBUG: Atualizando session_id para '{new_session_id}'")
                
                if st.session_state.current_chat_index < len(st.session_state.chat_history):
                    st.session_state.chat_history[st.session_state.current_chat_index]["id"] = new_session_id
                    print(f"DEBUG: Histórico atualizado com session_id '{new_session_id}'")
            
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": assistant_message, 
                "time": timestamp
            })
            
            if is_first_message:
                new_title = extract_title_from_response(assistant_message)
                st.session_state.chat_title = new_title
                
                if st.session_state.current_chat_index < len(st.session_state.chat_history):
                    st.session_state.chat_history[st.session_state.current_chat_index]["title"] = new_title
            
        typing_placeholder.empty()

# --- FIM DA FUNÇÃO PRINCIPAL DE LÓGICA DE MENSAGEM ---


def add_javascript1(): # Esta função é um duplicata de add_javascript, mantive para consistência
    """Adiciona JavaScript para melhorar a interação do usuário com o chat"""
    js_code = """
    <script>
    // Fazer com que a tecla Enter submeta o formulário
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            const textarea = document.querySelector('textarea');
            if (textarea) {
                textarea.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        const sendButton = document.querySelector('button[data-testid="baseButton-secondary"]');
                        if (sendButton) {
                            sendButton.click();
                        }
                    }
                });
            }
            
            // Mostrar o nome do arquivo quando for anexado
            const fileUploader = document.querySelector('.stFileUploader');
            if (fileUploader) {
                const observer = new MutationObserver(function(mutations) {
                    mutations.forEach(function(mutation) {
                        if (mutation.type === 'childList' && mutation.addedNodes.length) {
                            const fileInfo = fileUploader.querySelector('.uploadedFileName');
                            if (fileInfo) {
                                const fileName = fileInfo.textContent.trim();
                                // Adicionando uma mensagem ao lado do input
                                const inputContainer = document.querySelector('.input-container');
                                let fileStatus = document.querySelector('.file-attached');
                                
                                if (!fileStatus) {
                                    fileStatus = document.createElement('div');
                                    fileStatus.className = 'file-attached';
                                    inputContainer.insertBefore(fileStatus, inputContainer.firstChild);
                                }
                                
                                fileStatus.innerHTML = '<i class="fas fa-paperclip"></i> ' + fileName;
                            }
                        }
                    });
                });
                
                observer.observe(fileUploader, { childList: true, subtree: true });
            }
        }, 1000); // Pequeno atraso para garantir que os elementos foram carregados
    });
    </script>
    """
    st.components.v1.html(js_code, height=0)

def extract_title_from_response(response_text):
    """
    Extrai um título resumido da primeira resposta do assistente.
    """
    cleaned_text = re.sub(r'[\U00010000-\U0010ffff]|[\n\r]', '', response_text)
    
    sentences = re.split(r'\.', cleaned_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return f"Conversa ({datetime.now().strftime('%d/%m/%Y')})"
    
    sentence = sentences[1] if len(sentences) > 1 and len(sentences[0]) < 15 else sentences[0]
    
    if len(sentence) > 40:
        words = sentence.split()
        title = ""
        for word in words:
            if len(title) + len(word) + 1 <= 40:
                title += " " + word if title else word
            else:
                title += "..."
                break
    else:
        title = sentence
        
    if title and len(title) > 0:
        title = title[0].upper() + title[1:]
        
    return title

def regenerate_message(index):
    """Regenera a resposta a uma mensagem específica"""
    if index < 0 or index >= len(st.session_state.messages) or st.session_state.messages[index]["role"] != "user":
        return
    
    user_message_to_regenerate = st.session_state.messages[index]["content"]
    
    status_placeholder = st.empty()
    status_placeholder.info("Regenerando resposta...")
    
    with st.spinner():
        rag_context = get_rag_context()
        tech_context = "CONEXÃO DE DOMÍNIO TÉCNICO:\n- Redes e Firewalls\n- Segurança da Informação\n- Diagnóstico de Incidentes\n"
        final_context_for_model = f"{rag_context}\n{tech_context}" if rag_context else tech_context

        # O conversation_history para regeneração deve incluir as mensagens até a mensagem do usuário atual
        # e *excluir* a resposta do assistente que está sendo regenerada (se ela existir).
        # A forma mais segura é reconstruir o histórico até a mensagem do usuário em questão.
        
        # Copia o histórico ATÉ a mensagem do usuário que está sendo regenerada
        # Note: st.session_state.messages contém tanto user quanto assistant messages
        # generate_chat_prompt espera uma lista de mensagens.
        # Precisamos remover a resposta do assistente *anterior* se houver uma.
        
        # O histórico passado para generate_chat_prompt deve ser apenas os turnos ANTES do turno atual.
        # Se você está regenerando a resposta para messages[index] (que é um "user" role),
        # o histórico para generate_chat_prompt deve ser messages[0] a messages[index-1].
        
        conversation_history_for_regeneration = st.session_state.messages[:index]


        result = query_local_model(
            user_message_to_regenerate, 
            st.session_state.session_id, 
            context=final_context_for_model, 
            conversation_history=conversation_history_for_regeneration
        )
        
    if result:
        new_response = result.get('answer', 'Não foi possível regenerar a resposta.')
        
        timestamp = datetime.now().strftime("%H:%M")
        
        # Atualiza a mensagem do assistente existente ou insere uma nova
        if index + 1 < len(st.session_state.messages) and st.session_state.messages[index + 1]["role"] == "assistant":
            st.session_state.messages[index + 1] = {
                "role": "assistant",
                "content": new_response,
                "time": timestamp
            }
        else:
            # Caso não haja uma resposta do assistente diretamente após a mensagem do usuário
            st.session_state.messages.insert(index + 1, {
                "role": "assistant",
                "content": new_response,
                "time": timestamp
            })
    else:
        status_placeholder.error("Não foi possível regenerar a resposta. Por favor, tente novamente.")
        time.sleep(2) 
    
    status_placeholder.empty()
    st.rerun()


def edit_message(index, new_content):
    """Edita uma mensagem e regenera as respostas subsequentes"""
    if index < 0 or index >= len(st.session_state.messages):
        return
    
    st.session_state.messages[index]["content"] = new_content
    st.session_state.messages[index]["time"] = datetime.now().strftime("%H:%M") + " (editada)"
    
    st.session_state.editing_message = None # Sai do modo de edição
    
    if st.session_state.messages[index]["role"] == "user":
        # Se uma mensagem de usuário foi editada, regeneramos a resposta do assistente para ela.
        regenerate_message(index)
    else:
        # Se uma mensagem do assistente foi editada diretamente, apenas rerunda para atualizar a UI.
        st.rerun()


def create_new_chat():
    """Cria uma nova conversa"""
    st.session_state.session_id = ""
    st.session_state.messages = []
    st.session_state.chat_title = f"Nova Conversa ({datetime.now().strftime('%d/%m/%Y')})"
    
    st.session_state.chat_history.append({
        "id": "",
        "title": st.session_state.chat_title,
        "messages": []
    })
    
    st.session_state.current_chat_index = len(st.session_state.chat_history) - 1

def load_chat(index):
    """Carrega uma conversa existente"""
    st.session_state.current_chat_index = index
    chat = st.session_state.chat_history[index]
    st.session_state.session_id = chat["id"]
    st.session_state.messages = chat["messages"].copy()
    st.session_state.chat_title = chat["title"]
    st.rerun()

def delete_chat(index):
    """Exclui uma conversa"""
    if len(st.session_state.chat_history) > index:
        st.session_state.chat_history.pop(index)
        
        if not st.session_state.chat_history:
            create_new_chat()
        elif st.session_state.current_chat_index >= len(st.session_state.chat_history):
            st.session_state.current_chat_index = len(st.session_state.chat_history) - 1
            load_chat(st.session_state.current_chat_index)
        else:
            load_chat(st.session_state.current_chat_index)

def rename_chat():
    """Renomeia uma conversa existente"""
    if st.session_state.new_chat_title.strip():
        index = st.session_state.current_chat_index
        st.session_state.chat_history[index]["title"] = st.session_state.new_chat_title
        st.session_state.chat_title = st.session_state.new_chat_title
        
        st.session_state.renaming = False
        st.rerun()

st.markdown("""
    <style>
    /* Estilo Geral */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 0;
        max-width: 1200px;
    }
    
    /* Cabeçalho */
    .header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background-color: white;
        z-index: 999;
        padding: 1rem;
        border-bottom: 1px solid #e6e6e6;
    }
    
    /* Mensagens */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        align-self: flex-end;
    }
    
    .assistant-message {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 0.5rem;
        align-self: flex-start;
    }
    
    .message-time {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
        align-self: flex-end;
    }
    
    /* Entrada de mensagem */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 90%;
        max-width: 800px;
        background-color: white;
        padding: 1rem;
        border-top: 1px solid #e6e6e6;
        z-index: 998;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Botões */
    .primary-button {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    
    .secondary-button {
        border: 1px solid #4CAF50 !important;
        color: #4CAF50 !important;
    }
    
    .stButton button {
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Message Actions */
    .message-actions {
        display: flex;
        gap: 5px;
        margin-top: 5px;
        justify-content: flex-end;
    }
    
    .action-button {
        background-color: transparent;
        border: none;
        color: #4CAF50;
        cursor: pointer;
        font-size: 12px;
        padding: 2px 5px;
        border-radius: 3px;
    }
    
    .action-button:hover {
        background-color: rgba(76, 175, 80, 0.1);
    }
    
    /* Chat List */
    .chat-item {
        display: flex;
        align-items: center;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 0.25rem;
        cursor: pointer;
    }
    
    .chat-item:hover {
        background-color: rgba(76, 175, 80, 0.1);
    }
    
    .chat-item.active {
        background-color: rgba(76, 175, 80, 0.2);
        font-weight: bold;
    }
    
    .chat-item-title {
        flex-grow: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .chat-title {
        text-align: center;
        padding: 1rem;
        font-size: 1.5rem;
        font-weight: bold;
        color: #4CAF50;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: #4CAF50;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #45a049;
    }
    .attach-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 70px;
        color: #4CAF50;
        cursor: pointer;
        font-size: 20px;
    }
    
    /* Adicionando Font Awesome para o ícone */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css');
    
    /* Estilo para indicar quando um arquivo foi anexado */
    .file-attached {
        background-color: rgba(76, 175, 80, 0.1);
        border-radius: 4px;
        padding: 4px;
        margin-bottom: 5px;
        font-size: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .file-attached i {
        margin-right: 5px;
    }
    
    /* Estilo para o botão de remover anexo */
    .remove-file {
        color: #f44336;
        cursor: pointer;
    }
    
    /* Esconder elementos Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Editing message */
    .edit-message-container {
        display: flex;
        flex-direction: column;
        gap: 5px;
        margin-bottom: 10px;
    }
    
    .edit-actions {
        display: flex;
        justify-content: flex-end;
        gap: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# As funções handle_message_if_content e handle_message_with_input foram removidas,
# pois a lógica de envio é agora tratada por send_message_callback e _handle_message_logic.

if check_password():
    print("DEBUG AUTH: Verificando senha")
    if 'session_id' not in st.session_state:
        st.session_state.session_id = ""
        print("DEBUG: Inicializado session_id vazio")
    else:
        print(f"DEBUG: session_id existente: '{st.session_state.session_id}'")
        
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if 'current_chat_index' not in st.session_state:
        st.session_state.current_chat_index = 0
        
    if 'chat_title' not in st.session_state:
        st.session_state.chat_title = f"Nova Conversa ({datetime.now().strftime('%d/%m/%Y')})"
        
    if 'renaming' not in st.session_state:
        st.session_state.renaming = False
        
    if 'new_chat_title' not in st.session_state:
        st.session_state.new_chat_title = ""
        
    if 'editing_message' not in st.session_state:
        st.session_state.editing_message = None
        
    if 'edit_content' not in st.session_state:
        st.session_state.edit_content = ""
        
    if not st.session_state.chat_history:
        st.session_state.chat_history.append({
            "id": "",
            "title": st.session_state.chat_title,
            "messages": []
        })

    if 'use_rag' not in st.session_state:
        st.session_state.use_rag = False
    if 'rag_source' not in st.session_state:
        st.session_state.rag_source = "Texto Direto"
    if 'file_type' not in st.session_state:
        st.session_state.file_type = "TXT"
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'direct_text' not in st.session_state:
        st.session_state.direct_text = ""

    with st.sidebar:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(logo_path, width=50)
        with col2:
            st.markdown('<h2 style="margin-top: 0;">Chat IA</h2>', unsafe_allow_html=True)
        
        st.divider()
        
        st.button("🔄 Nova Conversa", on_click=create_new_chat, use_container_width=True)
        
        st.divider()
        
        st.markdown("### Minhas Conversas")
        for idx, chat in enumerate(st.session_state.chat_history):
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(f"📝 {chat['title']}", key=f"chat_{idx}", 
                            use_container_width=True,
                            help="Clique para abrir esta conversa"):
                    load_chat(idx)
            with col2:
                if st.button("🗑️", key=f"delete_{idx}", help="Excluir conversa"):
                    delete_chat(idx)
    
        use_rag = st.checkbox("Usar Contexto Adicional (RAG)", value=st.session_state.use_rag)
        st.session_state.use_rag = use_rag

        if use_rag:
            rag_source = st.radio(
                "Fonte do Contexto",
                ("Arquivo", "Texto Direto"),
                index=("Arquivo", "Texto Direto").index(st.session_state.rag_source)
            )
            st.session_state.rag_source = rag_source

            if rag_source == "Arquivo":
                file_type = st.selectbox(
                    "Tipo de Arquivo",
                    ("PDF", "TXT", "CSV"),
                    index=("PDF", "TXT", "CSV").index(st.session_state.file_type)
                )
                st.session_state.file_type = file_type
                uploaded_file = st.file_uploader(f"Carregar Arquivo {file_type}", type=file_type.lower(), key="file_uploader") # Use .lower() para type
                st.session_state.uploaded_file = uploaded_file

            elif rag_source == "Texto Direto":
                direct_text = st.text_area(
                    "Inserir Texto de Contexto", 
                    value=st.session_state.get('direct_text', ''),
                    height=150, 
                    key="direct_text"
                )
            st.divider()

            if st.button("Logout", use_container_width=True):
                logout()

    main_col1, main_col2, main_col3 = st.columns([1, 10, 1])
    
    with main_col2:
        add_javascript()
        if st.session_state.renaming:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text_input("Título da Conversa", value=st.session_state.chat_title, key="new_chat_title", label_visibility="collapsed")
            with col2:
                st.button("Salvar", on_click=rename_chat)
        else:
            col1, col2 = st.columns([10, 1])
            with col1:
                st.markdown(f'<div class="chat-title">{st.session_state.chat_title}</div>', unsafe_allow_html=True)
            with col2:
                if st.button("✏️", help="Renomear conversa"):
                    st.session_state.renaming = True
                    st.session_state.new_chat_title = st.session_state.chat_title
                    st.rerun()
        
        messages_container = st.container()
        
        st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)
        
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([5, 1, 1])

        with col1:
            st.text_area("Mensagem", placeholder="Digite sua mensagem aqui...", key="user_input", 
                height=70, label_visibility="collapsed")

        with col2:
            # Note: O key para o uploader de arquivo precisa ser único ou ser gerado de forma que permita resetá-lo
            # Adicionei um key gerado por uuid para permitir o reset.
            file_to_send = st.file_uploader("Anexar arquivo", type=["pdf", "txt", "csv", "doc", "docx", "xls", "xlsx"], 
                                        key=st.session_state.get('file_uploader_key', 'default_file_uploader_key'), label_visibility="collapsed")
            st.session_state.file_to_send = file_to_send # Armazena o arquivo no session_state
            
            st.markdown('<div class="attach-icon" title="Anexar arquivo"><i class="fas fa-paperclip"></i></div>', unsafe_allow_html=True)
            
            # Adicionar um pequeno display para o nome do arquivo anexado e um botão de remover
            if st.session_state.get('file_to_send', None) is not None:
                st.markdown(f"""
                    <div class="file-attached">
                        <i class="fas fa-paperclip"></i> {st.session_state.file_to_send.name} 
                        <span class="remove-file" onclick="window.parent.document.querySelector('button[data-testid*=\"file-uploader-delete-button\"]').click();">X</span>
                    </div>
                """, unsafe_allow_html=True)


        with col3:
            # O botão "Enviar" agora chama o callback
            st.button("Enviar", key="send_button", use_container_width=True, on_click=send_message_callback)
        
        with messages_container:
            for idx, message in enumerate(st.session_state.messages):
                if st.session_state.editing_message == idx:
                    st.markdown('<div class="edit-message-container">', unsafe_allow_html=True)
                    st.text_area("Editar mensagem", value=message["content"], key="edit_content", height=100)
                    
                    st.markdown('<div class="edit-actions">', unsafe_allow_html=True)
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("Cancelar", key=f"cancel_edit_{idx}"):
                            st.session_state.editing_message = None
                            st.rerun()
                    with col2:
                        if st.button("Salvar", key=f"save_edit_{idx}"):
                            edit_message(idx, st.session_state.edit_content)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    continue
                
                elif message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                        st.markdown(f"<div class='message-time'>{message['time']}</div>", unsafe_allow_html=True)
                        
                        st.markdown('<div class="message-actions">', unsafe_allow_html=True)
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.button("Editar", key=f"edit_{idx}"):
                                st.session_state.editing_message = idx
                                st.session_state.edit_content = message["content"]
                                st.rerun()
                        with col2:
                            if idx+1 < len(st.session_state.messages) and message["role"] == "user":
                                if st.button("Regenerar", key=f"regen_{idx}"):
                                    regenerate_message(idx)
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    with st.chat_message("assistant", avatar=logo_path):
                        st.write(message["content"])
                        st.markdown(f"<div class='message-time'>{message['time']}</div>", unsafe_allow_html=True)