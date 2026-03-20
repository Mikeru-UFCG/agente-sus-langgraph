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

# Exibe as mensagens antigas na tela (com memória de documentos)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Se houver documentos salvos nesta mensagem (rota RAG), mostra o expansor
        if message.get("docs"):
            with st.expander("📄 Ver trechos recuperados (Transparência de Fontes)"):
                for i, doc in enumerate(message["docs"]):
                    st.markdown(f"**Trecho {i+1}:** {doc.page_content}")
                    st.caption(f"📌 **Fonte:** {doc.metadata.get('source', 'Desconhecida')} | **Pág:** {doc.metadata.get('page', 'N/A')}")
                    st.divider()

# Caixa de texto para o usuário digitar
if prompt := st.chat_input("Faça uma pergunta sobre o SUS ou peça uma triagem..."):
    
    # 1. Mostra a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Chama o seu Sistema Agêntico (LangGraph)
    with st.chat_message("assistant"):
        with st.spinner("Processando... (O supervisor está avaliando sua rota 🧠)"):
            try:
                # Envia a pergunta para o grafo
                resultado = agent_app.invoke({"question": prompt})
                
                # Extrai os dados do estado final do LangGraph
                resposta = resultado["generation"]
                documentos = resultado.get("documents", [])
                rota = resultado.get("route", "Desconhecida")
                
                # --- TURBINA 1: Indicador visual da rota escolhida ---
                if rota == "RAG":
                    st.caption("🧭 **Rota do Supervisor:** Consulta a Documentos Oficiais (RAG)")
                elif rota == "AUTOMACAO":
                    st.caption("⚙️ **Rota do Supervisor:** Automação via MCP Local")

                # Mostra a resposta gerada pelo modelo
                st.markdown(resposta)
                
                # --- TURBINA 2: Expansor de transparência das fontes (Apenas para RAG) ---
                if documentos and rota == "RAG":
                    with st.expander("📄 Ver trechos recuperados (Transparência de Fontes)"):
                        for i, doc in enumerate(documentos):
                            st.markdown(f"**Trecho {i+1}:** {doc.page_content}")
                            st.caption(f"📌 **Fonte:** {doc.metadata.get('source', 'Desconhecida')} | **Pág:** {doc.metadata.get('page', 'N/A')}")
                            st.divider()
                
                # --- TURBINA 3: Salva no histórico incluindo os documentos ---
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": resposta,
                    "docs": documentos if rota == "RAG" else None
                })
                
            except Exception as e:
                st.error(f"Ocorreu um erro no processamento: {e}")