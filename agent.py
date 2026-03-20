import os
from typing import List, TypedDict

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from mcp_server import LocalMCPServer


class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    route: str


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"A variável de ambiente {name} não está definida. "
            f"Defina antes de executar. Ex.: export {name}='sua_chave'"
        )
    return value


# Garante que a chave exista sem hardcode no código
get_required_env("GOOGLE_API_KEY")

mcp = LocalMCPServer()

# Modelo mais estável para uso geral
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


def safe_extract_text(res) -> str:
    """
    Extrai texto da resposta do modelo de forma robusta.
    """
    content = getattr(res, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "".join(parts).strip()

    return str(content).strip()


def supervisor(state: GraphState):
    print("--- NÓ: SUPERVISOR ---")
    question = state["question"]

    prompt = f"""
Você é um roteador de intenções.

Analise a entrada abaixo:
{question}

Regras:
- Se o usuário pedir para triar, agendar ou registrar, responda apenas: AUTOMACAO
- Se for uma dúvida sobre saúde pública, SUS, atendimento, vacinação, princípios, diretrizes ou organização do SUS, responda apenas: RAG

Responda com uma única palavra.
""".strip()

    raw_res = llm.invoke(prompt)
    response = safe_extract_text(raw_res).upper()

    route = "AUTOMACAO" if "AUTOMACAO" in response else "RAG"
    return {"route": route}


def retrieve_node(state: GraphState):
    print("--- NÓ: RETRIEVER ---")
    question = state["question"]
    docs = retriever.invoke(question)
    return {"documents": docs}


def generate_node(state: GraphState):
    print("--- NÓ: ANSWERER / WRITER ---")

    documents = state.get("documents", [])

    contexto = "\n\n".join(
        [
            (
                f"Fonte: {d.metadata.get('source', 'desconhecida')} | "
                f"Página: {d.metadata.get('page', 'N/A')}\n"
                f"{d.page_content}"
            )
            for d in documents
        ]
    )

    prompt = f"""
Você é um assistente sobre o SUS.

Responda apenas com base no contexto recuperado.

Regras:
- Responda de forma objetiva e curta, em no máximo 5 linhas.
- Não invente informações.
- Se o contexto não bastar, diga: Não há informação suficiente nas fontes recuperadas.
- Cite a fonte e a página ao final, quando disponível.
- Não repita trechos longos do contexto.
- Finalize com:
Aviso: Este é um sistema automatizado. Não substitui consulta médica.

Contexto:
{contexto}

Pergunta:
{state['question']}
""".strip()

    raw_res = llm.invoke(prompt)
    return {"generation": safe_extract_text(raw_res)}


def self_check_node(state: GraphState):
    print("--- NÓ: SELF-CHECK ---")
    generation = state["generation"]

    prompt = f"""
Verifique a resposta abaixo.

Critérios:
- Possui aviso final informando que não substitui consulta médica
- Possui referência de fonte ou deixa claro quando não há base suficiente

Se estiver adequada, responda apenas: SIM
Se não estiver adequada, responda apenas: NAO

Resposta:
{generation}
""".strip()

    raw_res = llm.invoke(prompt)
    check = safe_extract_text(raw_res).upper()

    if "NAO" in check or "NÃO" in check:
        return {
            "generation": (
                "RECUSA: O sistema não encontrou fontes seguras ou formato adequado para responder.\n\n"
                "Aviso: Este é um sistema automatizado. Não substitui consulta médica."
            )
        }

    return {"generation": generation}


def automation_node(state: GraphState):
    print("--- NÓ: AUTOMATION ---")
    conteudo = f"PROTOCOLO DE TRIAGEM: {state['question']}\nStatus: Encaminhado."
    resultado_mcp = mcp.executar_tool(
        "salvar_arquivo_triagem",
        "nova_triagem.txt",
        conteudo,
    )
    return {"generation": f"Automação acionada. Servidor MCP diz: {resultado_mcp}"}


workflow = StateGraph(GraphState)

workflow.add_node("supervisor", supervisor)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("automation", automation_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["route"],
    {
        "RAG": "retrieve",
        "AUTOMACAO": "automation",
    },
)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
workflow.add_edge("automation", END)

app = workflow.compile()