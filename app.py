import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Environment variables setup
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Integrated Q&A and RAG Chatbot"

# Simple Q&A Prompt Template
simple_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user query concisely and accurately."),
    ("user", "Question: {question}")
])

# Function to generate simple Q&A response
def generate_simple_response(question, llm):
    output_parser = StrOutputParser()
    chain = simple_qa_prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

# Streamlit app setup
st.title("Integrated Q&A and Document-Based Chatbot")
st.write("Ask general questions or upload PDFs to query their content with chat history.")

# Sidebar for settings
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
llm_model = st.sidebar.selectbox("Select LLM", [
    "Gemma2-9b-It",
    "Deepseek-R1-Distill-Llama-70b",
    "Compound-Beta",
    "Llama-3.3-70b-Versatile",
    "Mistral-Saba-24b"
])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 150)

# Initialize session state for chat history
if 'store' not in st.session_state:
    st.session_state.store = {}

# Main interface
mode = st.radio("Select Mode", ["General Q&A", "Document Q&A"])

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model=llm_model, temperature=temperature, max_tokens=max_tokens)

    if mode == "General Q&A":
        st.write("## General Q&A")
        user_input = st.text_input("Enter your question here", key="general_qa_input")
        if user_input:
            response = generate_simple_response(user_input, llm)
            st.write("**Assistant:**", response)

    elif mode == "Document Q&A":
        st.write("## Document Q&A with Chat History")
        session_id = st.text_input("Session ID", value=f"session_{uuid.uuid4()}")
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            documents = []
            for uploaded_file in uploaded_files:
                temppdf = f"./temp_{uploaded_file.name}"
                with open(temppdf, "wb") as file:
                    file.write(uploaded_file.getvalue())
                loader = PyPDFLoader(temppdf)
                docs = loader.load()
                documents.extend(docs)
                os.remove(temppdf)  # Clean up temp file

            # Split and create embeddings
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(splits, embeddings)
            base_retriever = vectorstore.as_retriever()

            # Contextualize question prompt
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question, "
                "which might reference context in the chat history, "
                "formulate a standalone question that can be understood "
                "without the chat history. DO NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            history_aware_retriever = create_history_aware_retriever(
                llm, base_retriever, contextualize_q_prompt
            )

            # Answer question prompt
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. "
                "Use three sentences or less to answer the question.\n\n"
                "{context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            # Create RAG chain
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = RunnableMap({
                "context": history_aware_retriever,
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"]
            }) | question_answer_chain

            # Session history management
            def get_session_history(session: str) -> BaseChatMessageHistory:
                if session not in st.session_state.store:
                    st.session_state.store[session] = ChatMessageHistory()
                return st.session_state.store[session]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            user_input = st.text_input("Enter your question about the documents", key="doc_qa_input")
            if user_input:
                session_history = get_session_history(session_id)
                config = {
                    "configurable": {"session_id": session_id},
                    "chat_history": session_history.messages
                }
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config=config
                )
                st.write("**Assistant:**", response)
                st.write("**Chat History:**", session_history.messages)
        else:
            st.write("Please upload PDF files to start querying.")
else:
    st.warning("Please enter your Groq API key to use the chatbot.")
