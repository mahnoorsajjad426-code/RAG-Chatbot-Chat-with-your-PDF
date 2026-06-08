# 📄 RAG Chatbot — Chat with your PDF

A RAG (Retrieval Augmented Generation) chatbot that lets you upload any PDF and ask questions about its content. Built with Python, Streamlit, and LLaMA 3.3 via Groq.

## 🧠 How it works
1. Upload a PDF
2. The app extracts and chunks the text
3. Each chunk is converted into vector embeddings using SentenceTransformers
4. Your question is embedded and matched against the chunks using FAISS
5. The most relevant chunks are sent to LLaMA 3.3 which generates an answer

## 🛠️ Tech Stack
- **Streamlit** — Web UI
- **PyPDF2** — PDF text extraction
- **SentenceTransformers** — Local embeddings (all-MiniLM-L6-v2)
- **FAISS** — Vector similarity search
- **Groq API** — LLaMA 3.3 70B for answer generation
- **Python-dotenv** — Environment variable management

## 🚀 Run Locally

1. Clone the repo
```
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot
```

2. Install dependencies
```
   pip install -r requirements.txt
```

3. Create a `.env` file
```
   GROQ_API_KEY=your_groq_api_key_here
```

4. Run the app
```
   streamlit run app.py
```

## 👨‍💻 Author
Built by Ahsan — BS Artificial Intelligence, 4th Semester
