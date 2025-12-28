from langgraph.graph import StateGraph, END

from app.agents.retriever import retriever_agent
from app.agents.generator import generator_agent
from app.agents.validator import validator_agent
from app.agents.responder import responder_agent
from app.state import AgentState

def build_graph(retriever, llm):
    graph = StateGraph(AgentState)
    
    graph.add_node("retrieve", lambda state: retriever_agent(state, retriever))
    graph.add_node("generate", lambda state: generator_agent(state, llm))
    graph.add_node("validate", lambda state: validator_agent(state, llm))
    graph.add_node("respond", responder_agent)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")

    def routing_logic(state):
        if state["validated"]:
            return "respond"
        if state.get("retries",0)< 3:
            return "generate"
        return "respond"

    graph.add_conditional_edges(
        "validate",
        routing_logic,
        {
            "respond": "respond",
            "generate": "generate"
        }
    )

    graph.add_edge("respond", END)

    return graph.compile()