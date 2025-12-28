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
    prompt_text = f"""Based on the following context, answer the user's question. 
If the context doesn't contain enough information to answer, say so.

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