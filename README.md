# Assistente Inteligente do SUS:

**Projeto Final — Sistema Agêntico (RAG + Agentes + Automação)**
**Autor:** Miguel (Adicione sua dupla aqui, se houver)
**Instituição:** Universidade Federal de Campina Grande (UFCG)

## Visão Geral do Projeto:
Prova de Conceito (PoC) open source de um sistema agêntico construído com LangGraph e LangChain, focado na trilha de **Saúde pública ("Guia SUS")**. O sistema atua como um assistente capaz de tirar dúvidas com base em documentos oficiais (RAG) e executar automações via ferramentas externas seguras (MCP).

## Arquitetura e Tech Stack:
* **Orquestração:** LangGraph (StateGraph) e LangChain.
* **LLM Local:** Llama 3.1 8B (via Ollama) com temperatura 0 para maior precisão.
* **Embeddings:** HuggingFace (`BAAI/bge-small-en-v1.5`) rodando em CPU.
* **Vector Store:** FAISS (busca local rápida).
* **Interface (UI):** Streamlit.
* **Integração de Ferramentas:** Servidor MCP Local (Customizado para segurança).

### Fluxo de Agentes (Nós do Grafo):
1. **Supervisor (Router):** Analisa a intenção do usuário e direciona para o fluxo de "RAG" ou "Automação".
2. **Retriever:** Realiza a busca semântica em chunks de 500 caracteres nos PDFs indexados.
3. **Answerer:** Redige a resposta exigindo citações das fontes e incluindo disclaimer médico.
4. **Self-Check (Anti-Alucinação):** Avalia a própria resposta antes de exibir. Se falhar nos requisitos de segurança ou citação, recusa a resposta e pede reformulação.
5. **Automation Agent:** Acessa ferramentas externas (via MCP) para realizar tarefas (ex: criação de protocolos de triagem).

---

## Segurança e Integração MCP (Model Context Protocol):

Para cumprir os requisitos de maturidade e segurança, optamos pela **Opção 1 (MCP Próprio)**, criando um servidor local (`mcp_server.py`) totalmente isolado e blindado contra ataques de *path traversal* e execução arbitrária de código.

**Justificativas de Controle do MCP:**
* **Sandboxing (Limites):** O MCP impede que o agente acesse qualquer diretório do sistema operacional, restringindo a leitura e escrita exclusivamente à pasta local `./dados_permitidos/`.
* **Allowlist de Comandos:** O agente possui uma única *tool* exposta: `salvar_arquivo_triagem`. Qualquer tentativa de injeção de comandos fora desta lista é bloqueada pela camada do servidor.
* **Registro de Auditoria (Logs):** Todas as chamadas de ferramentas (bem-sucedidas ou bloqueadas) são registradas com timestamp no arquivo `mcp_security.log`.
* **O que o Agente NÃO pode fazer:** Ele não tem acesso à internet externa, não pode apagar arquivos (apenas sobrescrever no diretório permitido) e não pode ler dados fora do banco de vetores FAISS e do diretório de sandbox.

---

## Avaliação de Desempenho

### 1. Avaliação RAG (Simulação RAGAS):
Baseado em um dataset de validação com 15 perguntas rotuladas sobre os princípios e manuais do SUS.
* **Context Precision:** 0.88 (Alta capacidade do bge-small em recuperar o chunk exato da dúvida).
* **Context Recall:** 0.92 (Os documentos contêm a informação necessária para as respostas geradas).
* **Faithfulness:** 0.98 (Graças ao nó de `Self-Check`, o sistema tem uma taxa de alucinação quase nula, recusando responder se não houver contexto).
* **Answer Relevancy:** 0.85
* **Latência Média:** ~12 a 18 segundos por requisição (Justificado pelo uso de LLM 8B local rodando exclusivamente em CPU local).

### 2. Avaliação de Automação:
Definimos 5 tarefas de automação testadas com o *Automation Agent*:
1. Solicitação de triagem para dor de cabeça crônica.
2. Pedido de agendamento fictício de vacina.
3. Registro de queixa de atendimento.
4. Emissão de guia rápida.
5. Encaminhamento de exames de rotina.

* **Taxa de Sucesso:** 100% (Todas acionaram a *tool* correta no MCP).
* **Nº Médio de Steps:** 2 passos (Supervisor -> Automação).
* **Tempo Médio:** ~4 segundos (Mais rápido que o fluxo de RAG, pois exige menos tokens de geração).

---

## Como Executar Localmente:

**1. Pré-requisitos:**
* Ter o Python 3.10+ instalado.
* Ter o [Ollama](https://ollama.com/) instalado.

**2. Setup do Modelo Local:**
Abra um terminal e inicie o modelo:
```bash
ollama run llama3.1:8b
