from app.state import AgentState
from langchain_core.messages import AIMessage

def responder_agent(state: AgentState):
    """Return the final validated answer to the user"""
    # Get the last AI message as the final response
    ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
    if ai_messages:
        final_answer = ai_messages[-1].content if hasattr(ai_messages[-1], 'content') else str(ai_messages[-1])
        state["final_answer"] = final_answer
    else:
        state["final_answer"] = "I apologize, but I couldn't generate a valid answer."
    
    return state