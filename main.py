import streamlit as st
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

st.set_page_config(page_title="PDF Assistant")
st.title("PDF Assistant")

# load LLM
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=256,
        do_sample=True,
        temperature=0.3,
        device_map="auto")
    return pipe

llm = load_llm()

# embedding model
@st.cache_resource
def load_embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embedding_model = load_embedding_model()

# read pdf
def read_pdf(uploaded_file):
    pdf_text = ""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        pdf_text += page.get_text()
    return pdf_text

# retrieved doc
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# chat input with upload button
result = st.chat_input(
    "Upload document and ask a question",
    accept_file=True,
    file_type=["pdf"])

if result:
    question = result.text
    uploaded_file = result.files[0] if result.files else None

    # pdf processing
    if uploaded_file:
        file_id = uploaded_file.name + str(uploaded_file.size)
        if st.session_state.get("file_id") != file_id:
            with st.spinner("Reading and indexing PDF"):
                pdf_text = read_pdf(uploaded_file)
                docs = [Document(page_content=pdf_text)]
                splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
                st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
                st.session_state.file_id = file_id
            st.success(f"'{uploaded_file.name}' loaded")

    # answer question if vectorstore exists
    if question and "vectorstore" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # retriever chunks
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        context = format_docs(retriever.invoke(question))

        # system prompt
        prompt = f"""<|system|>
You are a helpful assistant answering questions about a PDF document.
Use the provided context to answer the question.</s>
<|user|>
Context:
{context}
Question: {question}</s>
<|assistant|>
"""
        with st.spinner("Thinking..."):
            response = llm(prompt)
            answer = response[0]["generated_text"]
            answer = answer.split("<|assistant|>")[-1].strip()

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

    elif question and "vectorstore" not in st.session_state:
        st.warning("Please upload a PDF alongside your question")