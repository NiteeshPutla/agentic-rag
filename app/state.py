from typing import TypedDict, List, Optional
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    retries: int
    validated: bool
    final_answer: Optional[str]
    