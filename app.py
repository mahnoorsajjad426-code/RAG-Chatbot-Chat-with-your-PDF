import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from rag_engine import extract_text_from_pdf, chunk_text, build_vector_store, retrieve_relevant_chunks, ask_groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app_state = {"chunks": None, "index": None}

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Chatbot</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-900 text-white min-h-screen flex flex-col items-center justify-center p-6">
        <div class="w-full max-w-xl bg-gray-800 p-8 rounded-xl shadow-xl space-y-6">
            <h1 class="text-2xl font-bold text-center text-indigo-400">📄 PDF RAG Chatbot</h1>
            <div class="border border-dashed border-gray-600 rounded p-4 text-center">
                <input type="file" id="pdfFile" accept=".pdf" class="mb-2">
                <button onclick="uploadPDF()" id="upBtn" class="bg-indigo-600 px-4 py-2 rounded font-bold block mx-auto mt-2">Process PDF</button>
            </div>
            <p id="status" class="text-center text-sm text-gray-400"></p>
            <div id="qa" class="opacity-50 pointer-events-none space-y-2">
                <input type="text" id="quest" placeholder="Ask a question..." class="w-full p-3 bg-gray-700 rounded border border-gray-600 text-white">
                <button onclick="ask()" id="askBtn" class="w-full bg-green-600 py-2 rounded font-bold">Ask</button>
                <div id="ansBox" class="hidden p-4 bg-gray-750 border border-gray-700 rounded mt-4">
                    <p class="text-indigo-400 font-bold">Answer:</p>
                    <p id="ans" class="text-gray-200 whitespace-pre-line"></p>
                </div>
            </div>
        </div>
        <script>
            async function uploadPDF() {
                const file = document.getElementById('pdfFile').files[0];
                if (!file) return;
                const fd = new FormData(); fd.append('file', file);
                document.getElementById('status').innerText = 'Processing PDF...';
                const res = await fetch('/api/upload', { method: 'POST', body: fd });
                const data = await res.json();
                if(res.ok) {
                    document.getElementById('status').innerText = data.message;
                    document.getElementById('qa').classList.remove('opacity-50', 'pointer-events-none');
                } else { document.getElementById('status').innerText = 'Error processing file.'; }
            }
            async function ask() {
                const q = document.getElementById('quest').value;
                if(!q) return;
                document.getElementById('askBtn').innerText = 'Thinking...';
                const fd = new FormData(); fd.append('question', q);
                const res = await fetch('/api/ask', { method: 'POST', body: fd });
                const data = await res.json();
                if(res.ok) {
                    document.getElementById('ans').textContent = data.answer;
                    document.getElementById('ansBox').classList.remove('hidden');
                }
                document.getElementById('askBtn').innerText = 'Ask';
            }
        </script>
    </body>
    </html>
    """

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_bytes = await file.read()
    text = extract_text_from_pdf(io.BytesIO(file_bytes))
    chunks = chunk_text(text)
    index, _ = build_vector_store(chunks)
    app_state["chunks"] = chunks
    app_state["index"] = index
    return {"message": "PDF uploaded and ready!"}

@app.post("/api/ask")
async def ask_question(question: str = Form(...)):
    if not app_state["index"]:
        raise HTTPException(status_code=400, detail="Upload a PDF first.")
    relevant_chunks = retrieve_relevant_chunks(question, app_state["index"], app_state["chunks"])
    answer = ask_groq(question, relevant_chunks)
    return {"answer": answer}
