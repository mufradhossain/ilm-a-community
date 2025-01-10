import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory

llm = None

def get_llm_instance(api_key):
    """
    Method to return the instance of llm model globally
    :return:
    """
    global llm
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            stream=True,
            google_api_key=api_key,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
    return llm

def get_response(user_query, conversation_history, api_key, system_prompt):
    """
    Method to return the response using the streaming chain
    :param user_query:
    :param conversation_history:
    :param system_prompt:
    :return:
    """
    prompt_template = f"""
    {system_prompt}
    
    Chat history: {conversation_history}
    
    User question: {user_query}
    """
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    llm = get_llm_instance(api_key)
    expression_language_chain = prompt | llm | StrOutputParser()

    # note: use .invoke() method for non-streaming
    return expression_language_chain.stream(
        {
            "conversation_history": conversation_history,
            "user_query": user_query
        }
    )

# let's  create the streamlit app
st.set_page_config(page_title="ILM-A Chatbot", page_icon=":robot:")
st.title("ILM-A Chatbot")

# Sidebar for API key input
api_key = st.sidebar.text_input("Enter your Gemini API key", type="password")

# Button to clear conversation and reset session state
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = [
        AIMessage(content="Hello, I am ILM-A! I am here to help you with your studies.")
    ]

# File selection mechanism
base_dir = "./books"
classes = os.listdir(base_dir)
selected_class = st.sidebar.selectbox("Select Class", classes)

subjects = os.listdir(os.path.join(base_dir, selected_class))
selected_subject = st.sidebar.selectbox("Select Subject", subjects)

chapters = os.listdir(os.path.join(base_dir, selected_class, selected_subject))
selected_chapter = st.sidebar.selectbox("Select Chapter", chapters)

# Check if the selection has changed
if "prev_class" not in st.session_state:
    st.session_state.prev_class = selected_class
if "prev_subject" not in st.session_state:
    st.session_state.prev_subject = selected_subject
if "prev_chapter" not in st.session_state:
    st.session_state.prev_chapter = selected_chapter

if (selected_class != st.session_state.prev_class or
    selected_subject != st.session_state.prev_subject or
    selected_chapter != st.session_state.prev_chapter):
    st.session_state.messages = [
        AIMessage(content="Hello, I am ILM-A! I am here to help you with your studies.")
    ]
    st.session_state.prev_class = selected_class
    st.session_state.prev_subject = selected_subject
    st.session_state.prev_chapter = selected_chapter

# Read the selected text file with UTF-8 encoding
chapter_path = os.path.join(base_dir, selected_class, selected_subject, selected_chapter)
with open(chapter_path, "r", encoding="utf-8") as file:
    chapter_content = file.read()

# System prompt
system_prompt = f"""
    You are a helpful assistant tutor for a student. 
    Your task is to assist the student with their understanding using the given context.
    Don't give answers directly rather try to assist them with guiding hints.
    Your responses should be concise.
    The context for the chapter is as follows:
    
{chapter_content}
"""

# initialize the messages key in streamlit session to store message history
if "messages" not in st.session_state:
    # add greeting message to user
    st.session_state.messages = [
        AIMessage(content="Hello, I am ILM-A! I am here to help you with your studies.")
    ]

# if there are messages already in session, write them on app
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

prompt = st.chat_input("Ask ILM-A a question...")

if prompt is not None and prompt != "":
    if not api_key:
        st.error("API key is not set. Please enter your Gemini API key in the sidebar.")
    else:
        # add the message to chat message container
        if not isinstance(st.session_state.messages[-1], HumanMessage):
            st.session_state.messages.append(HumanMessage(content=prompt))
            # display to the streamlit application
            message = st.chat_message("user")
            message.write(f"{prompt}")

        # Use only the last 5 pairs of messages for generating the response
        last_5_pairs = st.session_state.messages[-10:]

        if not isinstance(st.session_state.messages[-1], AIMessage):
            with st.chat_message("assistant"):
                # use .write() method for non-streaming, which means .invoke() method in chain
                response = st.write_stream(get_response(prompt, last_5_pairs, api_key, system_prompt))
            st.session_state.messages.append(AIMessage(content=response))
