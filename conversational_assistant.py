import streamlit as st
from google import genai
from google.genai import types
import json

restaurants = json.load(open("restaurants.json"))
dishes = json.load(open("dishes.json"))


SYSTEM_INSTRUCTION = f"""
Your name is Tunde, A conversational assistant for localbuka. your goal is to take a user's free-text message and respond helpfully using the data provided.
here is the available restaurants data
{restaurants}

here is the available dishes data:
{dishes}
"""

# Page configuration
st.set_page_config(page_title="LocalBuka AI Chat", page_icon="🥘", layout="centered")
st.title("🥘 LocalBuka Chat Assistant Prototype")
st.caption("A locabuka conversational chatbot powered by the gemini-2.5-flash model.")


# --- Sidebar for Authentication ---
with st.sidebar:
    st.header("Authentication")
    # First check if the key is already in environment variables, else ask the user
    api_key = st.text_input("Enter your Gemini API Key:", type="password", help="Get your key from Google AI Studio")
    st.markdown("[Get a Google AI Studio API Key](https://aistudio.google.com/)")

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


client = None

if api_key:
    client = genai.Client(api_key=api_key)
else:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass


# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render Existing Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("Type your message here..."):
    
    # Block input if client isn't configured yet
    if not client:
        st.error("Please provide a Gemini API Key in the sidebar or set it as an environment variable to chat.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Load the full chat history in the format required by the GenAI SDK
    formatted_history = []
    for msg in st.session_state.messages:
        role_map = "user" if msg["role"] == "user" else "model"
        formatted_history.append(
            types.Content(
                role=role_map,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # Streaming the Assistant Response from Gemini
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # We initialize the chat session with history.
            chat_config = types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION
            )

            chat = client.chats.create(
                model="gemini-2.5-flash",
                config=chat_config,
                history=formatted_history[:-1] # Pass all history except the newest one.
            )
            
            response_stream = chat.send_message_stream(prompt)
            
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌")
                
            response_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            full_response = "Sorry, I ran into an error generating that response."
            response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})