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
    generate_chat_prompt,
    invoke_local_model,
    get_model_context_size,
    search_knowledge_base
)

def add_javascript():
    """Adiciona JavaScript para a tecla Enter enviar a mensagem."""
    js_code = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            const textarea = document.querySelector('textarea[data-testid="stTextArea"]');
            if (textarea) {
                textarea.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        const sendButton = Array.from(document.querySelectorAll('button')).find(btn => btn.innerText.trim() === 'Enviar');
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
   page_icon="🤖",
   layout="wide",
   initial_sidebar_state="expanded"
)

logo_path = "logo.png"

def query_local_model(message, session_id="", model_params=None, context="", conversation_history=None):
    """Envia uma mensagem para o modelo local."""
    try:
        messages_for_model = generate_chat_prompt(message, conversation_history=conversation_history, context=context)
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
    """Retorna `True` se o usuário tiver a senha correta."""
    def password_entered():
        if hmac.compare_digest(st.session_state.get("username", "").strip(), "admin") and \
           hmac.compare_digest(st.session_state.get("password", "").strip(), "admin123"):
            st.session_state["password_correct"] = True
            if "password" in st.session_state: del st.session_state["password"]
            if "username" in st.session_state: del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False
            st.session_state["login_attempt"] = True

    if st.session_state.get("password_correct", False):
        return True

    st.markdown("""
        <style>
            .login-form { max-width: 400px; margin: 50px auto; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .login-title { text-align: center; }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="login-form">', unsafe_allow_html=True)
        st.markdown('<h1 class="login-title">Login</h1>', unsafe_allow_html=True)
        st.text_input("Usuário", key="username")
        st.text_input("Senha", type="password", key="password")
        st.button("Entrar", on_click=password_entered, use_container_width=True)

        if st.session_state.get("login_attempt", False) and not st.session_state.get("password_correct", False):
            st.error("Usuário ou senha incorretos")

        st.markdown('</div>', unsafe_allow_html=True)
    return False

def logout():
    """Faz logout limpando o estado da sessão de forma segura."""
    keys_to_delete = [
        "password_correct", "login_attempt", "username",
        "password", "auth_cookie"
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# CORREÇÃO: A função agora aceita um argumento para a busca.
def get_rag_context(user_query: str):
    """
    Obtém o contexto da base de conhecimento se a opção estiver ativa.
    """
    if st.session_state.get('use_rag', False):
        # A função chama a busca usando a pergunta do usuário como query.
        return search_knowledge_base(user_query)
    return ""

def handle_message(user_message_content):
    """Processa a mensagem, busca na base de conhecimento e chama o modelo."""
    if not user_message_content or not user_message_content.strip():
        return

    is_first_message = not st.session_state.messages
    user_message_raw = " ".join(user_message_content.strip().split())

    st.session_state.messages.append({"role": "user", "content": user_message_raw, "time": datetime.now().strftime("%H:%M")})

    with st.chat_message("assistant", avatar=logo_path):
        typing_placeholder = st.empty()
        typing_placeholder.markdown("... 🤔")

        # CORREÇÃO: A chamada para get_rag_context agora está correta.
        rag_context = get_rag_context(user_message_raw)
        history_for_model = st.session_state.messages[:-1]

        result = query_local_model(
            user_message_raw,
            st.session_state.session_id,
            context=rag_context,
            conversation_history=history_for_model
        )

    typing_placeholder.empty()

    assistant_message = result.get('answer', 'Não foi possível obter uma resposta.')
    st.session_state.session_id = result.get("sessionId", st.session_state.session_id)

    st.session_state.last_prompt_tokens = result.get("prompt_tokens", 0)
    st.session_state.last_completion_tokens = result.get("completion_tokens", 0)
    st.session_state.last_total_tokens = result.get("total_tokens", 0)

    st.session_state.messages.append({"role": "assistant", "content": assistant_message, "time": datetime.now().strftime("%H:%M")})

    if is_first_message and 'extract_title_from_response' in globals():
        new_title = extract_title_from_response(assistant_message)
        st.session_state.chat_title = new_title
        if st.session_state.current_chat_index != -1:
            st.session_state.chat_history[st.session_state.current_chat_index]["title"] = new_title
            st.session_state.chat_history[st.session_state.current_chat_index]["id"] = st.session_state.session_id

    if st.session_state.current_chat_index != -1:
        st.session_state.chat_history[st.session_state.current_chat_index]["messages"] = st.session_state.messages

    st.rerun()

def extract_title_from_response(response_text):
    """Extrai um título resumido da primeira resposta."""
    first_line = response_text.split('\n')[0]
    words = first_line.split()
    title = ' '.join(words[:6])
    if len(words) > 6:
        title += "..."
    return title if title else "Nova Conversa"

def regenerate_message(index):
    """Regenera a resposta a uma mensagem de usuário."""
    if index < 0 or index >= len(st.session_state.messages) or st.session_state.messages[index]["role"] != "user":
        return

    user_message_to_regenerate = st.session_state.messages[index]["content"]
    history_for_regeneration = st.session_state.messages[:index]

    with st.spinner("Regenerando resposta..."):
        # CORREÇÃO: Passa a pergunta do usuário para get_rag_context
        rag_context = get_rag_context(user_message_to_regenerate)
        result = query_local_model(
            user_message_to_regenerate,
            st.session_state.session_id,
            context=rag_context,
            conversation_history=history_for_regeneration
        )

    new_response = result.get('answer', 'Não foi possível regenerar a resposta.')
    timestamp = datetime.now().strftime("%H:%M")

    if index + 1 < len(st.session_state.messages):
        st.session_state.messages[index + 1] = {"role": "assistant", "content": new_response, "time": timestamp}
    else:
        st.session_state.messages.append({"role": "assistant", "content": new_response, "time": timestamp})

    st.rerun()

def edit_message(index, new_content):
    """Edita uma mensagem e regenera a resposta se for do usuário."""
    st.session_state.messages[index]["content"] = new_content
    st.session_state.editing_message = None

    if st.session_state.messages[index]["role"] == "user":
        if index + 1 < len(st.session_state.messages) and st.session_state.messages[index + 1]["role"] == "assistant":
            st.session_state.messages.pop(index + 1)
        regenerate_message(index)
    else:
        st.rerun()

def create_new_chat():
    """Cria uma nova conversa, salvando a anterior se necessário."""
    if st.session_state.current_chat_index != -1 and st.session_state.messages:
        st.session_state.chat_history[st.session_state.current_chat_index]["messages"] = st.session_state.messages.copy()

    new_chat_title = f"Nova Conversa ({datetime.now().strftime('%d/%m/%Y')})"
    st.session_state.chat_history.append({
        "id": "", "title": new_chat_title, "messages": []
    })

    st.session_state.current_chat_index = len(st.session_state.chat_history) - 1

    st.session_state.messages = []
    st.session_state.session_id = ""
    st.session_state.chat_title = new_chat_title
    st.rerun()

def load_chat(index):
    """Carrega uma conversa existente do histórico."""
    if st.session_state.current_chat_index != -1 and st.session_state.messages:
        st.session_state.chat_history[st.session_state.current_chat_index]["messages"] = st.session_state.messages.copy()

    st.session_state.current_chat_index = index
    chat = st.session_state.chat_history[index]
    st.session_state.messages = chat["messages"].copy()
    st.session_state.session_id = chat["id"]
    st.session_state.chat_title = chat["title"]
    st.rerun()

def delete_chat(index):
    """Exclui uma conversa do histórico."""
    if 0 <= index < len(st.session_state.chat_history):
        st.session_state.chat_history.pop(index)

        if st.session_state.current_chat_index == index:
            if st.session_state.chat_history:
                load_chat(len(st.session_state.chat_history) - 1)
            else:
                create_new_chat()
        elif st.session_state.current_chat_index > index:
             st.session_state.current_chat_index -= 1
             st.rerun()
        else:
            st.rerun()


if not check_password():
    st.stop()

# Bloco de inicialização do st.session_state
defaults = {
    'session_id': "", 'messages': [], 'chat_history': [], 'current_chat_index': -1,
    'chat_title': "Nova Conversa", 'editing_message': None, 'edit_content': '',
    'use_rag': False, 'rag_source': 'Texto Direto', 'file_type': 'PDF',
    'uploaded_file': None, 'direct_text': '', 'last_prompt_tokens': 0,
    'last_completion_tokens': 0, 'last_total_tokens': 0
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if 'model_context_size' not in st.session_state:
    st.session_state.model_context_size = get_model_context_size()

if not st.session_state.chat_history:
    create_new_chat()

# --- Sidebar ---
with st.sidebar:
    st.image(logo_path, width=50)
    st.title("Modelo de IA")
    if st.button("🔄 Nova Conversa", use_container_width=True):
        create_new_chat()
    st.divider()

    st.markdown("### Minhas Conversas")
    for i in reversed(range(len(st.session_state.chat_history))):
        chat = st.session_state.chat_history[i]
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(f"{chat['title']}", key=f"chat_{i}", use_container_width=True, help="Abrir esta conversa"):
                load_chat(i)
        with col2:
            if st.button("🗑️", key=f"delete_{i}", help="Excluir conversa"):
                delete_chat(i)

    st.divider()
    # CORREÇÃO: Lógica do RAG na sidebar simplificada para apenas uma checkbox.
    st.session_state.use_rag = st.checkbox(
        "💡 Usar Base de Conhecimento",
        value=st.session_state.get('use_rag', False),
        help="Permite que o modelo consulte os documentos pré-processados para obter respostas mais precisas."
    )

    st.divider()
    with st.expander("📊 Métricas da Última Interação"):
        st.markdown(f"**Entrada (Prompt):** `{st.session_state.last_prompt_tokens}` tokens")
        st.markdown(f"**Saída (Resposta):** `{st.session_state.last_completion_tokens}` tokens")
        st.markdown(f"**Total:** `{st.session_state.last_total_tokens}` tokens")

        context_size = st.session_state.model_context_size
        if context_size > 0:
            percentual_uso = (st.session_state.last_total_tokens / context_size) * 100
            st.progress(percentual_uso / 100, text=f"Uso do Contexto: {percentual_uso:.1f}% de {context_size}")

    st.divider()
    if st.button("Logout", use_container_width=True):
        logout()

# --- Interface Principal do Chat ---
st.header(st.session_state.chat_title)

chat_container = st.container(height=500, border=True)
with chat_container:
    for idx, message in enumerate(st.session_state.messages):
        avatar = logo_path if message["role"] == "assistant" else "user"
        with st.chat_message(message["role"], avatar=avatar):
            if st.session_state.editing_message == idx:
                new_content = st.text_area("Editar:", value=message["content"], key=f"edit_content_{idx}")
                col1, col2 = st.columns(2)
                if col1.button("Salvar", key=f"save_{idx}"):
                    edit_message(idx, new_content)
                if col2.button("Cancelar", key=f"cancel_{idx}"):
                    st.session_state.editing_message = None
                    st.rerun()
            else:
                st.markdown(message["content"])
                if message["role"] == "user":
                    col_b1, col_b2, col_b_spacer = st.columns([1, 1, 5])
                    with col_b1:
                        if st.button("✏️ Editar", key=f"edit_{idx}", help="Editar sua mensagem"):
                            st.session_state.editing_message = idx
                            st.rerun()
                    with col_b2:
                        if st.button("🔄 Regenerar", key=f"regen_{idx}", help="Gerar nova resposta"):
                            regenerate_message(idx)

# --- Entrada de Mensagem ---
user_input = st.chat_input("Digite sua mensagem aqui...")
if user_input:
    handle_message(user_input)

add_javascript()
