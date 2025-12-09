import os
import time
from typing import TypedDict, Literal, List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Imports locais
from src.PCA import ManualPCA
from src.RAG import build_vector_store

# Configuração e Chaves
load_dotenv()
chave_api = os.getenv("GEMINI_API_KEY")
if not chave_api: raise ValueError("Chave ausente!")

# Modelos LLM
llm_padrao = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=chave_api)
llm_analista = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, google_api_key=chave_api)

# Inicializa Ferramentas
print("--- 🚀 Init Ferramentas ---")
retriever = build_vector_store("data/movies.csv", chave_api)
pca_tool = ManualPCA("data/ratings.csv", "data/movies.csv")
pca_tool.fit(k=20) 

# --- TRIAGEM ---
class TriagemFilmes(BaseModel):
    intencao: Literal["RECOMENDAR", "HISTORICO_USUARIO", "INFO_GERAL", "PEDIR_INFO"]
    user_id: Optional[int] = Field(description="ID numérico do usuário.")
    tema: Optional[str] = Field(description="Tema ou contexto.")

TRIAGEM_PROMPT = """
Analise o pedido:
- **RECOMENDAR**: Sugestões novas.
- **HISTORICO_USUARIO**: Perguntas sobre perfil/passado (gostos, estatísticas).
- **INFO_GERAL**: Fatos gerais de filmes.
- **PEDIR_INFO**: Falta ID.
Retorne JSON.
"""

# --- ESTADO ---
class AgentState(TypedDict):
    pergunta: str
    triagem: dict
    resposta_final: str

# --- NÓS ---
def node_triagem(state: AgentState):
    print(f"\n--- 🚦 Triagem: '{state['pergunta']}' ---")
    res = llm_padrao.with_structured_output(TriagemFilmes).invoke([
        SystemMessage(content=TRIAGEM_PROMPT), HumanMessage(content=state['pergunta'])
    ])
    return {"triagem": res.model_dump()}

def node_recomendacao(state: AgentState):
    print("--- 🔢 Math Tool: Recomendando ---")
    time.sleep(2)
    u_id = state["triagem"].get("user_id")
    
    # Chama ferramenta (Retorna apenas a lista agora)
    rec_list = pca_tool.recommend(user_id=u_id)
    
    prompt = f"O usuário {u_id} pediu recomendações. Apresente esta lista de forma simpática:\n{rec_list}"
    res = llm_padrao.invoke(prompt)
    return {"resposta_final": res.content}

def node_historico(state: AgentState):
    print("--- 🕵️ Data Analyst: Histórico ---")
    time.sleep(2)
    u_id = state["triagem"].get("user_id")
    
    raw_data = pca_tool.get_user_raw_data(user_id=u_id, limit=50)
    if not raw_data: return {"resposta_final": "Usuário não encontrado."}

    # Prompt Analista
    prompt = f"""
    Como Analista de Cinema, responda à pergunta: "{state['pergunta']}"
    Baseie-se nestes DADOS REAIS do Usuário {u_id}:
    {raw_data}
    Seja analítico e direto.
    """
    res = llm_analista.invoke(prompt)
    return {"resposta_final": res.content}

def node_info_rag(state: AgentState):
    print("--- 📚 RAG Tool: Contexto ---")
    time.sleep(1)
    docs = retriever.invoke(state['pergunta'])
    ctx = "\n".join([d.page_content for d in docs])
    res = llm_padrao.invoke(f"Contexto: {ctx}\nPergunta: {state['pergunta']}")
    return {"resposta_final": res.content}

def node_pedir_id(state: AgentState):
    return {"resposta_final": "Preciso do ID numérico (ex: 1, 10)."}

# --- FLUXO ---
def decidir(state: AgentState):
    i = state["triagem"]["intencao"]
    uid = state["triagem"].get("user_id")
    
    if i == "RECOMENDAR": return "math_rec" if uid else "ask"
    elif i == "HISTORICO_USUARIO": return "math_hist" if uid else "ask"
    elif i == "INFO_GERAL": return "rag"
    return "ask"

wf = StateGraph(AgentState)
wf.add_node("triagem", node_triagem)
wf.add_node("calculadora", node_recomendacao)
wf.add_node("analista", node_historico)
wf.add_node("rag", node_info_rag)
wf.add_node("ask", node_pedir_id)

wf.add_edge(START, "triagem")
wf.add_conditional_edges("triagem", decidir, {
    "math_rec": "calculadora", "math_hist": "analista", "rag": "rag", "ask": "ask"
})
wf.add_edge("calculadora", END)
wf.add_edge("analista", END)
wf.add_edge("rag", END)
wf.add_edge("ask", END)

app_graph = wf.compile()