import streamlit as st
from agent import app as agent_app

# Configuração da página
st.set_page_config(page_title="Assistente SUS", page_icon="🏥", layout="centered")

# Título e descrição
st.title("🏥 Assistente Inteligente do SUS")
st.markdown("""
**Projeto Final - Sistema Agêntico (RAG + Automação)**
Este assistente pode responder dúvidas sobre o SUS com base em documentos oficiais ou realizar protocolos automatizados de triagem.
""")
st.divider()

# Inicializa o histórico de chat na sessão do Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe as mensagens antigas na tela
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Caixa de texto para o usuário digitar
if prompt := st.chat_input("Faça uma pergunta sobre o SUS ou peça uma triagem..."):
    
    # 1. Mostra a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Chama o seu Sistema Agêntico (LangGraph)
    with st.chat_message("assistant"):
        with st.spinner("Processando... (O supervisor está avaliando sua rota)"):
            try:
                # Envia a pergunta para o grafo que fizemos no agent.py
                resultado = agent_app.invoke({"question": prompt})
                resposta = resultado["generation"]
                
                # Mostra a resposta na tela
                st.markdown(resposta)
                
                # Salva no histórico
                st.session_state.messages.append({"role": "assistant", "content": resposta})
            except Exception as e:
                st.error(f"Ocorreu um erro no processamento: {e}")