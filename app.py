import streamlit as st
import sqlite3
import uuid
import time
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load API key
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

# Set up the Gemini 1.5 Pro model
llm = GoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-1.5-pro")

# Initialize SQLite database
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS chat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    role TEXT,
    content TEXT
)
""")
conn.commit()

# Function to save messages
def save_message(session_id, role, content):
    cursor.execute("INSERT INTO chat (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
    conn.commit()

# Function to load chat history
def load_chat_history(session_id):
    cursor.execute("SELECT role, content FROM chat WHERE session_id = ?", (session_id,))
    return cursor.fetchall()

# Chat history instance
def chat_history(session_id):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat_history.db"
    )

# Generate unique session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Custom CSS for UI enhancements
st.markdown("""
    <style>
        body {
            background-color: #F0F8FF;
        }
        .title-text {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: linear-gradient(45deg, #FF5733, #1E88E5);
            padding: 15px;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
            animation: fadeIn 2s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stTextInput {
            position: fixed;
            bottom: 10px;
            width: 80%;
            left: 10%;
            z-index: 999;
            border-radius: 20px;
            padding: 10px;
            border: 2px solid #1E88E5;
        }
        .chat-container {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 3px 3px 12px rgba(0,0,0,0.2);
        }
        .user-message {
            color: #00897B;
            font-weight: bold;
        }
        .assistant-message {
            color: #D81B60;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Display title with animation
st.markdown("""
<h1 class='title-text'>✨💬 AI Data Science Tutor 🚀✨</h1>
""", unsafe_allow_html=True)

# New Chat Button with emoji
theme_button = st.button("🔄 Start a New Chat")
if theme_button:
    st.session_state.session_id = str(uuid.uuid4())  # Generate new session
    st.session_state.messages = []  # Clear chat history
    st.rerun()  # Refresh the app

# Get session ID
session_id = st.session_state.session_id
chat_history_instance = chat_history(session_id)

# Define Chat Prompt Template
chat_prompt = ChatPromptTemplate(
    messages=[
        ('system', """You are an AI assistant specialized in Data Science tutoring. 
                      You will only answer questions related to Data Science. 
                      If asked anything outside this topic, politely decline and request a Data Science-related question.
                   """),
        MessagesPlaceholder(variable_name="history", optional=True),
        ('human', '{prompt}')
    ]
)

# Define output parser
out_parser = StrOutputParser()

# Create a chain
chain = chat_prompt | llm | out_parser

# Define Runnable with message history
chat = RunnableWithMessageHistory(
    chain,
    lambda session: SQLChatMessageHistory(session, "sqlite:///chat_history.db"),
    input_messages_key="prompt",
    history_messages_key="history"
)

# Chat History Container
st.markdown("### 📜 Chat History")
chat_container = st.container()

# Load chat history and display it
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history(session_id)

with chat_container:
    for role, content in st.session_state.messages:
        with st.chat_message(role):
            if role == "user":
                st.markdown(f"<p class='user-message'>👤 {content}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='assistant-message'>🤖 {content}</p>", unsafe_allow_html=True)

# User input at the bottom
user_input = st.text_input("💡 Type your message here:", key="user_message")

# If user submits a message
if user_input:
    save_message(session_id, "user", user_input)
    st.session_state.messages.append(("user", user_input))

    # Invoke AI model
    config = {'configurable': {'session_id': session_id}}
    response = chat.invoke({'prompt': user_input}, config)

    save_message(session_id, "assistant", response)
    st.session_state.messages.append(("assistant", response))

    # Display AI response with animation
    with chat_container:
        with st.chat_message("assistant"):
            st.markdown(f"<p class='assistant-message'>🤖 {response}</p>", unsafe_allow_html=True)

    # Clear the input field
    st.session_state.pop("user_message")
    st.session_state["user_message"] = ""
    st.rerun()  # Refresh the app
