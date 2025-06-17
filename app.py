import streamlit as st
import os

# pip install streamlit pinecone langchain_pinecone langchain_cohere langchain-azure-ai langchain
# azure-ai-inference langchain-azure-ai langchain-core

# Langchain Imports
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureOpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore as Pinecone_Langchain
from langchain_cohere import CohereRerank
from langchain.chat_models import init_chat_model

# ======================================================================================
# 1. LOGIN AUTHENTICATION (This now runs first to act as a gate)
# ======================================================================================

st.set_page_config(page_title="Reeeliance Internal Chatbot", layout="wide")

# In app.py, replace the old check_password function with this one.

def check_password():
    """Returns `True` if the user has the correct password, `False` otherwise."""

    # If the user is already authenticated, just return True.
    if st.session_state.get("password_correct", False):
        return True

    # --- Use columns for a centered layout ---
    col1, col2, col3 = st.columns([1, 1.5, 1])

    with col2:
        # --- 1. Add a professional header ---
        st.image("reeeliance-logo-rgb@1x.png") # Optional: Uncomment if you have a logo
        st.title("Reeeliance Internal Chatbot")
        st.markdown("Please enter your credentials to continue.")

        # --- 2. Use a form for a better UX ---
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            # The "Log In" button is the form's submit button.
            submitted = st.form_submit_button("Log In")

            # --- 3. Check credentials ONLY on form submission ---
            if submitted:
                # Compare submitted credentials with those from st.secrets
                correct_username = st.secrets["APP_USER"]
                correct_password = st.secrets["APP_PASSWORD"]

                if username == correct_username and password == correct_password:
                    st.session_state["password_correct"] = True
                    # Rerun the script to remove the login form and show the app
                    st.rerun()
                else:
                    # If login fails, show an error message.
                    st.error("😕 Username or password incorrect")
    
    # If we are here, it means the user is not yet authenticated.
    return False


if not check_password():
    st.stop()  # Do not continue if the password is not correct.

# --- If the password is correct, the rest of the app will run. ---


# ======================================================================================
# 2. RAG CHAIN AND APP SETUP (Only runs after successful login)
# ======================================================================================

# --- Caching RAG Chain Setup ---
@st.cache_resource
def setup_rag_chain():
    """Initializes all the components for the RAG chain."""
    st.info("Initializing RAG chain... This happens once per session.")

    # Directly access the secret as an environment variable
    PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']

    # Cohere
    cohere_rerank = CohereRerank(cohere_api_key=st.secrets['COHERE_API_KEY'], model="rerank-english-v3.0")

    # Azure configurations
    os.environ["AZURE_INFERENCE_ENDPOINT"] = st.secrets["AZURE_INFERENCE_ENDPOINT"]
    os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI_ENDPOINT"]
    os.environ["AZURE_INFERENCE_CREDENTIAL"] = st.secrets["AZURE_INFERENCE_CREDENTIAL"]
    os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_INFERENCE_CREDENTIAL"]

    # Initialize the LLM
    azure_llm = init_chat_model(model="gpt-4o", model_provider="azure_ai")

    # Initialize the OpenAI embeddings
    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")

    # Initialize Pinecone
    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    index = pinecone.Index(st.secrets['PINECONE_INDEX_NAME'])
    vectorstore = Pinecone_Langchain(index, embeddings, 'text')

    # Initialize Cohere reranker
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=cohere_rerank, base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10})
    )

    # Final Retriever Chain Prompt
    template = """
                You are a friendly and helpful assistant. Your main task is to answer the user's question based on the provided context.

                Key Instructions:
                1. Base your answer on the provided context. You may use your own knowledge to make the response more natural and complete, but you must not contradict the facts in the context.
                2. If the context does not contain enough information to answer the question, clearly state that you couldn't find the answer in the provided documents. Do not try to make up an answer.
                3. Always answer in the same language as the user's question.

                Context:
                {context}

                Question:
                {question}

                Answer:
                """
    prompt = PromptTemplate.from_template(template)

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=azure_llm,
        retriever=compression_retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        chain_type='stuff',
        verbose=True
    )
    return conversational_chain

# --- UI and App Logic ---

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    if "rag_chat_history" in st.session_state:
        del st.session_state.rag_chat_history
    st.rerun()

# --- Sidebar Layout ---
with st.sidebar:
    st.title("🔎 Reeeliance Chatbot")
    st.markdown("---")
    st.markdown("""
    Welcome to the reeeliance internal chatbot! Use this tool to ask questions or get technical assistance.
    """)
    st.markdown("---")
    
    selected_model = st.selectbox(
        "Select Model",
        ["RAG - Internal Documents", "Chatbot (Not Implemented)"]
    )

    if st.button("Clear Chat History"):
        clear_chat_history()

    st.sidebar.markdown("""
    ---
    ⚠️ **Reminder**: Please clear chat history when not needed to save API costs.
    """)

# --- Main Chat Interface ---

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

if "rag_chat_history" not in st.session_state:
    st.session_state["rag_chat_history"] = []

st.title("💬 Reeeliance Internal Chatbot")
st.caption("Always verify responses before implementation - this tool provides suggestions only.")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("Enter your question here:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        if selected_model == "RAG - Internal Documents":
            with st.spinner("Searching internal documents and generating answer..."):
                rag_chain = setup_rag_chain()
                result = rag_chain({
                    "question": user_input,
                    "chat_history": st.session_state.rag_chat_history
                })
                full_response = result.get("answer", "Sorry, I could not find an answer.")
                response_placeholder.markdown(full_response)

                with st.expander("Show Sources"):
                    for doc in result.get("source_documents", []):
                        metadata = doc.metadata
                        source_file = metadata.get("file_name", "Unknown Source")
                        st.info(f"Source: {source_file}")

                st.session_state.rag_chat_history.append((user_input, full_response))
        else:
            full_response = "This chatbot model is not yet implemented."
            response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})