import streamlit as st
import time
from typing import Optional, List, Iterator
import requests
from datetime import datetime

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = "default_user"

# API configuration
API_URL = "http://localhost:8000"

def get_user_sessions():
    response = requests.get(f"{API_URL}/sessions/{st.session_state.user_id}")
    return response.json()["sessions"]

def load_chat_history(session_id: str):
    response = requests.get(
        f"{API_URL}/chat_history/{session_id}/{st.session_state.user_id}"
    )
    history = response.json()
    return history["messages"]

# Sidebar for session management
with st.sidebar:
    st.title("Chat Sessions")
    
    # Option to start new chat
    if st.button("Start New Chat"):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()
    
    # Session selection
    sessions = get_user_sessions()
    if sessions:
        st.write("Previous Sessions:")
        for session in sessions:
            if st.button(f"Session {session[:8]}...", key=session):
                st.session_state.session_id = session
                st.session_state.messages = load_chat_history(session)
                st.rerun()

# Main chat interface
st.title("AI Insurance Assistant")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        response = requests.post(
            f"{API_URL}/query",
            json={
                "text": prompt,
                "user_id": st.session_state.user_id,
                "session_id": st.session_state.session_id,
                "new_session": st.session_state.session_id is None
            }
        )
        
        if response.status_code == 200:
            response_data = response.json()
            full_response = response_data["content"]
            st.session_state.session_id = response_data["session_id"]
            
            message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })
        else:
            message_placeholder.error("Error: Failed to get response from the assistant.")