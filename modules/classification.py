from pydantic import BaseModel, Field
from typing import List

class TicketClassification(BaseModel):
    """Classify a support ticket with topic tags, sentiment, and priority."""
    topic_tags: List[str] = Field(..., description="Relevant tags like 'How-to', 'Product', 'Connector', 'Bug', 'API/SDK'.")
    sentiment: str = Field(..., description="The user's sentiment, e.g., 'Frustrated', 'Curious', 'Angry'.")
    priority: str = Field(..., description="The ticket priority, e.g., 'P0 (High)', 'P1 (Medium)', 'P2 (Low)'.")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

def get_classification_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(TicketClassification)
    system_prompt = "You are an expert at analyzing and classifying customer support tickets. Analyze the following ticket and classify it according to the provided schema. Your response must be in the specified structured format."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{ticket_text}")
    ])
    return prompt | structured_llm
