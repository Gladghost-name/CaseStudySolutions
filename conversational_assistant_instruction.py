import streamlit as st
from google import genai
from google.genai import types

# --- Hardcoded System Instruction ---
SYSTEM_INSTRUCTION = "Your name is adara gladness, you are a assistant for localbuka."

# Page configuration
st.set_page_config(page_title="Gemini 2.5 Flash Chat", page_icon="⚡", layout="centered")
st.title("⚡ Gemini 2.5 Flash Chat")
st.caption("A sleek, conversational chatbot powered by Google's high-performance gemini-2.5-flash model.")

# --- Sidebar for Settings & Authentication ---
with st.sidebar:
    st.header("Settings")
    
    # API Key Input
    api_key = st.text_input("Enter your Gemini API Key:", type="password", help="Get your key from Google AI Studio")
    st.markdown("[Get a Google AI Studio API Key](https://aistudio.google.com/)")
    
    st.divider()
    
    # Button to clear chat history
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Initialize Gemini Client ---
client = None
if api_key:
    client = genai.Client(api_key=api_key)
elif "GEMINI_API_KEY" in st.secrets:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

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

    # 1. Display user message immediately & append to session history
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Reconstruct the full chat history in the format required by the Google GenAI SDK
    formatted_history = []
    for msg in st.session_state.messages:
        role_map = "user" if msg["role"] == "user" else "model"
        formatted_history.append(
            types.Content(
                role=role_map,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # 3. Stream the Assistant Response from Gemini
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # Prepare configuration with the hardcoded System Instruction
            chat_config = types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION
            )

            # Initialize the chat session with history AND the config
            chat = client.chats.create(
                model="gemini-2.5-flash",
                config=chat_config,
                history=formatted_history[:-1] # Pass all history except the newest message
            )
            
            # Send the latest user message and stream the chunks
            response_stream = chat.send_message_stream(prompt)
            
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    # Live typing effect
                    response_placeholder.markdown(full_response + "▌")
                
            # Remove the cursor block when finished
            response_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            full_response = "Sorry, I ran into an error generating that response."
            response_placeholder.markdown(full_response)

    # 4. Save assistant response to state
    st.session_state.messages.append({"role": "assistant", "content": full_response})