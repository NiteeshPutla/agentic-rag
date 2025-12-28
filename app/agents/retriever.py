from langchain_core.messages import HumanMessage
from app.state import AgentState

def retriever_agent(state: AgentState, retriever):
    """Retrieve relevant documents from vector store based on user query"""
    if not state["messages"]:
        return state
    
    # Get the first user message (HumanMessage)
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not user_messages:
        return state
    
    query = user_messages[0].content if hasattr(user_messages[0], 'content') else str(user_messages[0])
    
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    state["documents"] = docs
    
    return state

