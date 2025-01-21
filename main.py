from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import uvicorn
from phi.model.groq import Groq
from phi.agent import Agent, RunResponse
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType
from phi.storage.agent.postgres import PgAgentStorage
from phi.embedder.google import GeminiEmbedder
from phi.tools.googlesearch import GoogleSearch
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import Json

load_dotenv()

class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime

class ChatHistory(BaseModel):
    messages: List[Message]
    session_id: str

class Query(BaseModel):
    text: str
    user_id: Optional[str] = "default_user"
    session_id: Optional[str] = None
    new_session: Optional[bool] = False

class AgentResponse(BaseModel):
    content: str
    session_id: str


# Database configuration
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
db_params = {
    "dbname": "ai",
    "user": "ai",
    "password": "ai",
    "host": "localhost",
    "port": "5532"
}

# Create messages table if it doesn't exist
def init_db():
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_session_user ON chat_messages(session_id, user_id);
    """)
    conn.commit()
    cur.close()
    conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database and load models/resources
    init_db()
    print("Database initialized")
    
    yield  # Server is running and handling requests
    
    # Cleanup: You can add any cleanup code here
    print("Shutting down")

app = FastAPI(lifespan=lifespan)


# Initialize knowledge base and storage components
knowledge_base = PDFKnowledgeBase(
    path='data/',
    vector_db=PgVector(
        table_name="news_paper",
        db_url=db_url,
        search_type=SearchType.vector,
        embedder=GeminiEmbedder(dimensions=300),
    ),       
)
# knowledge_base.load(upsert=True, recreate=True)
storage = PgAgentStorage(table_name="News_paper_vect", db_url=db_url)

def store_message(session_id: str, user_id: str, role: str, content: str):
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO chat_messages (session_id, user_id, role, content)
        VALUES (%s, %s, %s, %s)
        """,
        (session_id, user_id, role, content)
    )
    conn.commit()
    cur.close()
    conn.close()

def get_chat_history(session_id: str, user_id: str) -> List[Dict]:
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT role, content, timestamp 
        FROM chat_messages 
        WHERE session_id = %s AND user_id = %s 
        ORDER BY timestamp
        """,
        (session_id, user_id)
    )
    messages = [
        {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        for role, content, timestamp in cur.fetchall()
    ]
    cur.close()
    conn.close()
    return messages

def create_agent(session_id: Optional[str] = None, user: str = "user") -> Agent:
    if not session_id:
        existing_sessions = storage.get_all_session_ids(user)
        if existing_sessions:
            session_id = existing_sessions[0]

    # web_agent = Agent(
    #     name='Web Agent',
    #     model=Groq(id='llama-3.3-70b-versatile'),
    #     tools=[GoogleSearch()],
    #     instructions=[
    #         'search for the first 5 results',
    #         'Try to match the the keywords to the results and provide your answer',
    #         'If there is no results just relay your thoughts'
    #     ],
    #     show_tool_calls=True,
    #     markdown=True   
    # )
    
    # knowledge_agent = Agent(
    #     model=Groq(id='llama3-70b-8192'),
    #     knowledge=knowledge_base,
    #     search_knowledge=True,
    #     markdown=True,
    #     instructions=[
    #         'Search the knowledge base extensively for the results'
    #     ]
    # )

    return Agent(
        model=Groq(id='llama3-70b-8192'),
        knowledge=knowledge_base,
        search_knowledge=True,
        markdown=True,
        instructions=[
            'Search the knowledge_base extensively for the results',
            "if the results is not found, start your reply with using my own idea"
        ]
    )

@app.post("/query", response_model=AgentResponse)
async def query_agent(query: Query):
    try:
        # Create agent instance
        agent = create_agent(session_id=query.session_id, user=query.user_id)
        
        # Store user message
        store_message(agent.session_id, query.user_id, "user", query.text)
        
        # Get response
        response: RunResponse = agent.run(query.text)
        
        # Store assistant message
        store_message(agent.session_id, query.user_id, "assistant", response.content)
        
        return AgentResponse(
            content=response.content,
            session_id=agent.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat_history/{session_id}/{user_id}", response_model=ChatHistory)
async def get_session_history(session_id: str, user_id: str):
    messages = get_chat_history(session_id, user_id)
    return ChatHistory(messages=messages, session_id=session_id)

@app.get("/sessions/{user_id}")
async def get_user_sessions(user_id: str):
    return {"sessions": storage.get_all_session_ids(user_id)}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "connected"}

# Initialize database on startup


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)