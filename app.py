import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent import graph

st.set_page_config(page_title="Literature Triage AI", page_icon="📚")
st.title("📚 Systematic Literature Review Agent")

if "GROQ_API_KEY" not in st.secrets or "OPENROUTER_API_KEY" not in st.secrets:
    st.error("API keys missing. Please configure .streamlit/secrets.toml")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if prompt := st.chat_input("Ask about your edge computing or computer vision PDFs..."):
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agents are synthesizing the literature..."):
            inputs = {"messages": st.session_state.messages}
            result = graph.invoke(inputs)

            final_response = result["messages"][-1].content
            st.markdown(final_response)
            st.session_state.messages.append(AIMessage(content=final_response))
