import os
from typing import List, TypedDict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

# ---> INTEGRAÇÃO COM O SERVIDOR MCP LOCAL
from mcp_server import LocalMCPServer
mcp = LocalMCPServer()

# 1. ESTADO DO GRAFO (A memória temporária do agente)
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    route: str

# 2. CONFIGURAÇÃO DO MODELO E BANCO DE DADOS
llm = ChatOllama(model="llama3.1:8b", temperature=0) # Mude para qwen2.5 se precisar
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. NÓS DO GRAFO (As "Pessoas" da sua equipe)

def supervisor(state: GraphState):
    """Decide se a pergunta vai pro RAG ou para Automação."""
    print("--- NÓ: SUPERVISOR ---")
    question = state["question"]
    prompt = f"""Você é um roteador. Analise a entrada do usuário: '{question}'.
    Se o usuário estiver pedindo para 'triar', 'agendar', 'registrar' ou 'automatizar', responda apenas AUTOMACAO.
    Se for uma dúvida sobre saúde, vacinas, ou SUS, responda apenas RAG.
    Responda APENAS com a palavra RAG ou AUTOMACAO."""
    
    response = llm.invoke(prompt).content.strip().upper()
    route = "AUTOMACAO" if "AUTOMACAO" in response else "RAG"
    return {"route": route}

def retrieve_node(state: GraphState):
    """Busca documentos no banco FAISS."""
    print("--- NÓ: RETRIEVER ---")
    question = state["question"]
    docs = retriever.invoke(question)
    return {"documents": docs}

def generate_node(state: GraphState):
    """Gera a resposta com citações e disclaimer de saúde."""
    print("--- NÓ: ANSWERER / WRITER ---")
    question = state["question"]
    docs = state["documents"]
    
    contexto = "\n\n".join([f"Trecho: {d.page_content}\nFonte: {d.metadata.get('source', 'Desconhecida')} - Pág: {d.metadata.get('page', 'N/A')}" for d in docs])
    
    prompt = f"""Você é um assistente de saúde do SUS. Responda à pergunta usando APENAS o contexto abaixo. 
    É OBRIGATÓRIO citar a fonte (nome do arquivo e página) no meio ou fim do seu texto.
    No final da resposta, adicione o seguinte AVISO OBRIGATÓRIO: 'Aviso: Este é um sistema automatizado. Não substitui consulta médica.'
    
    Contexto: {contexto}
    Pergunta: {question}
    Resposta:"""
    
    response = llm.invoke(prompt).content
    return {"generation": response}

def self_check_node(state: GraphState):
    """Mecanismo Anti-Alucinação (Exigência do Projeto)."""
    print("--- NÓ: SELF-CHECK (ANTI-ALUCINAÇÃO) ---")
    generation = state["generation"]
    
    prompt = f"""Analise a seguinte resposta gerada por um assistente:
    '{generation}'
    A resposta possui o aviso médico obrigatório E citações de fontes?
    Responda apenas SIM ou NAO."""
    
    check = llm.invoke(prompt).content.strip().upper()
    if "NAO" in check or "NÃO" in check:
        return {"generation": "RECUSA: O sistema falhou na verificação de segurança ou não encontrou fontes suficientes. Por favor, reformule a pergunta."}
    return {"generation": generation}

def automation_node(state: GraphState):
    """Agente de Automação: Usa a tool do MCP para executar o processo."""
    print("--- NÓ: AUTOMATION AGENT (VIA MCP) ---")
    question = state["question"]
    
    # Prepara o conteúdo
    conteudo = f"=== PROTOCOLO DE TRIAGEM ===\nDemanda original: {question}\nStatus: Encaminhado para a unidade básica de saúde.\n"
    
    # Em vez de usar open() direto, o agente CHAMA A FERRAMENTA DO MCP
    resultado_mcp = mcp.executar_tool(
        tool_name="salvar_arquivo_triagem", 
        nome_arquivo="nova_triagem.txt", 
        conteudo=conteudo
    )
    
    return {"generation": f"Automação acionada!\n\n**Resposta do Servidor MCP:**\n{resultado_mcp}"}

# 4. ORQUESTRAÇÃO (Ligando os pontos com LangGraph)

def route_decision(state: GraphState):
    return state["route"]

workflow = StateGraph(GraphState)

# Adicionando os nós
workflow.add_node("supervisor", supervisor)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("self_check", self_check_node)
workflow.add_node("automation", automation_node)

# Desenhando as arestas (caminhos)
workflow.set_entry_point("supervisor")

# Do supervisor, vai para RAG (Retrieve) ou Automação
workflow.add_conditional_edges(
    "supervisor",
    route_decision,
    {
        "RAG": "retrieve",
        "AUTOMACAO": "automation"
    }
)

# Caminho do RAG
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "self_check")
workflow.add_edge("self_check", END)

# Caminho da Automação
workflow.add_edge("automation", END)

# Compilando o aplicativo
app = workflow.compile()

# 5. TESTE LOCAL NO TERMINAL
if __name__ == "__main__":
    print("\n" + "="*50)
    print("SISTEMA AGÊNTICO SUS INICIADO")
    print("="*50)
    
    # Teste 1: RAG
    print("\n>>> TESTE 1: PERGUNTA DE SAÚDE (RAG)")
    inputs_rag = {"question": "Quais são os princípios do SUS?"}
    resultado_rag = app.invoke(inputs_rag)
    print(f"\nRESPOSTA FINAL:\n{resultado_rag['generation']}")
    
    # Teste 2: Automação
    print("\n" + "-"*50)
    print(">>> TESTE 2: PEDIDO DE AUTOMAÇÃO")
    inputs_auto = {"question": "Por favor, faça a triagem para mim pois estou com muita dor de cabeça."}
    resultado_auto = app.invoke(inputs_auto)
    print(f"\nRESPOSTA FINAL:\n{resultado_auto['generation']}")