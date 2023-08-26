import streamlit as st
import os
from convo import DocumentEmbedder  # Bringing in the DocumentEmbedder from the convo module

st.title("Chat-with-your-codebase")

# Collect user OpenAI API Key
user_key = st.text_input("Input your OpenAI API Key", "")
if user_key:
    os.environ['OPENAI_API_KEY'] = user_key

    # Collect GitHub repository URL
    user_repo = st.text_input("Provide the GitHub URL of your public codebase", "https://github.com/facebookresearch/segment-anything.git")
    if user_repo:
        st.write("Received URL:", user_repo)

        # Initialize DocumentEmbedder and clone the GitHub repo
        embedder = DocumentEmbedder(user_repo)
        embedder.clone_repository()
        st.write("Successfully cloned the repository")

        # Process and embed the codebase
        st.write("Embedding the codebase, please wait...")
        embedder.initialize_db()
        st.write("Embedding complete. Ready for your queries.")

        # Set up chat history if not already done
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Show previous chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Get new user query
        if prompt := st.chat_input("What would you like to ask?"):
            # Log user query to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Show user query in chat
            with st.chat_message("user"):
                st.markdown(prompt)
            # Generate and show assistant's answer
            response = embedder.fetch_answers(prompt)
            # Show assistant's answer in chat
            with st.chat_message("assistant"):
                st.markdown(response)
            # Log assistant's answer to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

