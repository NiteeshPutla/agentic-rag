from langchain_core.messages import AIMessage, HumanMessage
from app.state import AgentState

def validator_agent(state: AgentState, llm):
    """Validate if the generated answer is grounded in the context"""
    # Check retry limit
    if state.get("retries", 0) >= 3:
        state["validated"] = True  # Force validation after max retries
        return state
    
    if not state["messages"] or not state["documents"]:
        state["validated"] = False
        return state
    
    # Get the generated answer (last AI message)
    ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
    if not ai_messages:
        state["validated"] = False
        return state
    
    generated_answer = ai_messages[-1].content if hasattr(ai_messages[-1], 'content') else str(ai_messages[-1])
    
    # Get context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    
    # Construct validation prompt
    prompt_text = f"""You are a validator. Check if the following answer is grounded in the provided context.
Answer with only "Yes" or "No".

Context:
{context}

Generated Answer:
{generated_answer}

Is the answer grounded in the context? Answer Yes or No:"""
    
    # Get validation verdict using HumanMessage for chat models
    prompt_message = HumanMessage(content=prompt_text)
    verdict = llm.invoke([prompt_message])
    
    verdict_text = verdict.content.strip().lower() if hasattr(verdict, 'content') else str(verdict).strip().lower()
    
    is_valid = "yes" in verdict_text
    state["validated"] = is_valid

    # IMPROVEMENT: Add feedback to the message history if validation fails
    if not is_valid:
        feedback = "Validation failed: The previous answer was not fully grounded in the context. Please try again and ensure every claim is supported by the provided documents."
        state["messages"].append(AIMessage(content=feedback))
    
    return state