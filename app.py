import streamlit as st
from rag_engine import extract_text_from_pdf, chunk_text, build_vector_store, retrieve_relevant_chunks, ask_groq
st.set_page_config(page_title="RAG CHATBOT", page_icon="📄")
st.title("START WITH YOUR PDF")

# Upload PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    text = extract_text_from_pdf(uploaded_file)

    # Chunk the text
    chunks = chunk_text(text)

    # Build vector store
    index, embeddings = build_vector_store(chunks)

    st.success("PDF processed successfully! You can now ask questions about the content.")

    # Question input
    question = st.text_input("Ask a question about the PDF:")

    if question:
        with st.spinner("Thinking..."):
            # Step 4: Get relevant chunks
            relevant_chunks = retrieve_relevant_chunks(question, index, chunks)
            
            # Step 5: Ask Groq
            answer = ask_groq(question, relevant_chunks)
        
        st.markdown("### Answer:")
        st.write(answer)

