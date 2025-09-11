from pydantic import BaseModel, Field
from typing import List

class TicketClassification(BaseModel):
    """Classify a support ticket with comprehensive topic tags, sentiment, and priority analysis."""
    topic_tags: List[str] = Field(
        ..., 
        description="Relevant tags from: 'How-to', 'Product', 'Connector', 'Lineage', 'API/SDK', 'SSO', 'Glossary', 'Best practices', 'Sensitive data', 'Bug', 'Feature request', 'Documentation', 'Performance', 'Integration', 'Security', 'Onboarding', 'Troubleshooting'"
    )
    sentiment: str = Field(
        ..., 
        description="User's emotional state: 'Frustrated', 'Curious', 'Angry', 'Neutral', 'Satisfied', 'Confused', 'Urgent', 'Appreciative'"
    )
    priority: str = Field(
        ..., 
        description="Urgency level: 'P0 (High)' for blocking issues/angry customers, 'P1 (Medium)' for important but non-blocking, 'P2 (Low)' for general questions"
    )
    summary: str = Field(
        ...,
        description="Brief 1-2 sentence summary of the ticket's main issue or request"
    )
    suggested_action: str = Field(
        ...,
        description="Recommended next step or team assignment based on the classification"
    )

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

def get_classification_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(TicketClassification)
    
    system_prompt = """You are an expert customer support analyst specializing in data platform and analytics tools. 
    
    Analyze the following ticket and provide comprehensive classification:

    TOPIC TAGS - Choose 1-3 most relevant from:
    - How-to: Step-by-step guidance requests
    - Product: General product questions or feedback
    - Connector: Data source connection issues
    - Lineage: Data lineage and flow questions
    - API/SDK: Programming interface questions
    - SSO: Single sign-on and authentication
    - Glossary: Data dictionary and terminology
    - Best practices: Methodology and optimization
    - Sensitive data: Privacy and security concerns
    - Bug: System errors or malfunctions
    - Feature request: New capability suggestions
    - Documentation: Missing or unclear docs
    - Performance: Speed and efficiency issues
    - Integration: Third-party tool connections
    - Security: Access control and permissions
    - Onboarding: New user setup and training
    - Troubleshooting: Problem diagnosis help

    SENTIMENT - Assess the user's emotional state:
    - Frustrated: Repeated issues, blocked workflow
    - Angry: Strong negative emotions, escalation language
    - Curious: Learning-focused, exploratory questions
    - Neutral: Matter-of-fact, professional tone
    - Satisfied: Positive feedback or acknowledgment
    - Confused: Unclear about concepts or processes
    - Urgent: Time-sensitive, business-critical needs
    - Appreciative: Thankful, positive interaction

    PRIORITY - Business impact assessment:
    - P0 (High): Production issues, angry customers, security concerns, complete workflow blockage
    - P1 (Medium): Important features needed, significant delays, frustrated users
    - P2 (Low): General questions, nice-to-have features, documentation requests

    SUMMARY: Provide a concise overview of the main issue
    SUGGESTED_ACTION: Recommend next steps or team assignment
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Ticket Content:\n{ticket_text}")
    ])
    return prompt | structured_llm