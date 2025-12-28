from langchain_core.messages import HumanMessage, AIMessage
from app.state import AgentState

def generator_agent(state: AgentState, llm):
    """Generate answer based on retrieved context"""
    # OPTIMIZATION: Increment retry counter
    state["retries"] = state.get("retries", 0) + 1
    
    # Get user query (first HumanMessage)
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    user_query = user_messages[0].content if user_messages else ""
    
    # Build context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    
    # Construct prompt
    # Note: If this is a retry, the 'messages' list already contains the previous 
    # failed attempt and the validator's feedback, which helps the LLM self-correct.
    prompt_text = f"""You are a strict assistant. Answer the user's question using ONLY the context provided.

    If the user asks for information not present in the context, or asks for examples of what is missing, 
    simply state: "The provided documents do not contain this information." 
    Do NOT invent placeholder strings or identifiers to illustrate what is missing.
Context:
{context}

User Question: {user_query}

Answer:"""
    
    prompt_message = HumanMessage(content=prompt_text)
    
    # We pass the full message history so the LLM sees the feedback from the validator
    response = llm.invoke(state["messages"] + [prompt_message])
    
    # Add AI response to messages
    if isinstance(response, str):
        state["messages"].append(AIMessage(content=response))
    else:
        state["messages"].append(response)
    
    return state