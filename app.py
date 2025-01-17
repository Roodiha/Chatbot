import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from agent import agent_executor


st.title("Edu Help")
# chat interface for consistent queries
if "messages" not in st.session_state:
    st.session_state.messages = []

def prepare_chat_history(messages):
    chat_history = []
    for message, kind in messages:
        if kind == "ai":
            message = AIMessage(message)
        elif kind == "user":
            message = HumanMessage(message)
        chat_history.append(message)
    return chat_history

# Display for all the messages
for message, kind in st.session_state.messages:
    with st.chat_message(kind):
        st.markdown(message)

prompt = st.chat_input("Ask your questions ...")

if prompt:
    # Handling prompts and rendering to the chat interface
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append(
        [prompt, "user"]
    )

    with st.spinner("Generating response"):
        chat_history = prepare_chat_history(st.session_state.messages)
        output = agent_executor.invoke(
            {"input": prompt, "chat_history": chat_history}
        )["output"]
        
        st.chat_message("ai").markdown(output)
        st.session_state.messages.append([output, "ai"])