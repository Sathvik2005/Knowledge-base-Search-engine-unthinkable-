# 🚀 Answer IQ - Quick Start Guide

## Installation & Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Streamlit (Web UI)
- PyTorch (Deep Learning)
- Transformers (LLM Models)
- LangChain (RAG Framework)
- FAISS (Vector Search)
- OpenAI Whisper (Voice Transcription)
- And other dependencies

### Step 2: Run the Application

**Windows:**
```bash
run.bat
```

**Mac/Linux:**
```bash
chmod +x run.sh
./run.sh
```

**Or manually:**
```bash
streamlit run app.py
```

### Step 3: Access the Application

Open your browser and go to: **http://localhost:8501**

---

## 📖 How to Use

### 1. Upload Documents

1. Look at the left sidebar
2. Click **"Browse files"** under "Upload Documents"
3. Select one or more PDF or TXT files
4. Click **"🚀 Build Knowledge Base"**
5. Wait for processing to complete (you'll see a success message)

### 2. Ask Questions (Text)

1. Go to the **"💬 Text Query"** tab
2. Type your question in the text area
3. Optional: Check "Concise Mode" for shorter answers
4. Click **"🔍 Search"**
5. View your answer and source documents

**Example Questions:**
- "What is artificial intelligence?"
- "Explain machine learning"
- "What are the benefits of RAG?"
- "List the types of machine learning"

### 3. Ask Questions (Voice)

1. Go to the **"🎤 Voice Query"** tab
2. Click **"Browse files"** and upload an audio file (WAV, MP3, M4A, OGG)
3. Click **"🎧 Transcribe & Search"**
4. The system will:
   - Transcribe your audio to text
   - Search the knowledge base
   - Show the answer with sources

### 4. View History

1. Go to the **"📜 History"** tab
2. See all previous queries and answers
3. Click **"🔄 Refresh History"** to update
4. Expand any query to see full details

---

## 🎯 Tips for Best Results

### Document Preparation

✅ **DO:**
- Use clear, well-formatted documents
- Include multiple related documents for better context
- Use PDFs with selectable text (not scanned images)
- Keep documents focused on specific topics

❌ **DON'T:**
- Upload extremely large files (>10MB per file)
- Mix completely unrelated topics
- Use password-protected PDFs
- Upload scanned documents without OCR

### Asking Questions

✅ **DO:**
- Be specific and clear
- Ask one question at a time
- Use keywords from your documents
- Rephrase if you don't get good results

❌ **DON'T:**
- Ask multiple unrelated questions together
- Use overly vague questions
- Ask about information not in your documents
- Expect answers outside the knowledge base

### Voice Input

✅ **DO:**
- Speak clearly and at a moderate pace
- Use a quiet environment
- Record in high quality (WAV preferred)
- Keep recordings under 30 seconds

❌ **DON'T:**
- Use very noisy recordings
- Mumble or speak too fast
- Use extremely low-quality audio
- Record very long questions (>1 minute)

---

## ⚙️ Configuration Options

You can customize the system by editing `app.py`:

```python
CONFIG = {
    "chunk_size": 500,          # Size of text chunks (smaller = more precise)
    "chunk_overlap": 100,        # Overlap between chunks
    "retrieval_k": 5,            # Number of documents to retrieve
    "max_tokens": 300,           # Max length of generated answers
    "whisper_model": "base",     # Whisper model: tiny, base, small, medium, large
}
```

### Whisper Model Options

| Model  | Size | Speed | Accuracy | Use Case |
|--------|------|-------|----------|----------|
| tiny   | 39 MB | Fastest | Low | Quick testing |
| base   | 74 MB | Fast | Good | **Recommended** |
| small  | 244 MB | Medium | Better | Higher accuracy |
| medium | 769 MB | Slow | Best | Professional use |
| large  | 1550 MB | Slowest | Excellent | Maximum accuracy |

---

## 🐛 Troubleshooting

### Problem: "Out of Memory" Error

**Solution:**
1. Reduce `chunk_size` to 300
2. Reduce `retrieval_k` to 3
3. Upload fewer documents at once
4. Close other applications

### Problem: Slow Performance

**Solution:**
1. Use a smaller Whisper model (tiny or base)
2. Use GPU if available
3. Reduce number of uploaded documents
4. Reduce `max_tokens` to 150

### Problem: Whisper Not Working

**Solution:**
1. Install ffmpeg:
   - Windows: Download from https://ffmpeg.org/
   - Mac: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`
2. Ensure audio files are in supported formats
3. Try converting audio to WAV format

### Problem: Poor Answer Quality

**Solution:**
1. Upload more relevant documents
2. Increase `retrieval_k` to 7-10
3. Disable "Concise Mode"
4. Rephrase your question
5. Check if the answer is actually in your documents

### Problem: Can't Upload Documents

**Solution:**
1. Check file format (PDF or TXT only)
2. Ensure file is not corrupted
3. Try uploading one file at a time
4. Check file permissions

---

## 💡 Advanced Usage

### Custom Model Configuration

Replace models in `app.py`:

```python
# For better answers (requires more RAM):
"llm_model": "microsoft/DialoGPT-large"

# For faster processing:
"llm_model": "gpt2"

# For multilingual support:
"embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### Batch Processing

Process multiple queries at once:

1. Use the History feature to track results
2. Upload all documents first
3. Save queries in a text file
4. Process one by one

### Export Results

Query history is saved in `query_history.json`:

```json
[
  {
    "timestamp": "2026-01-30T12:00:00",
    "question": "What is AI?",
    "answer": "Artificial Intelligence is...",
    "num_sources": 5
  }
]
```

---

## 📊 Performance Expectations

### Processing Time

| Operation | CPU | GPU |
|-----------|-----|-----|
| Load 10 PDFs | 30-60s | 15-30s |
| Create embeddings | 20-40s | 5-15s |
| Initialize LLM | 10-20s | 5-10s |
| Answer query | 5-15s | 2-5s |
| Transcribe audio (30s) | 10-20s | 3-8s |

### Resource Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 5 GB free
- Internet: Only for initial model download

**Recommended:**
- CPU: 8+ cores or GPU
- RAM: 16 GB
- Storage: 10 GB free
- GPU: 4GB+ VRAM (optional but recommended)

---

## 🎓 Example Use Cases

### 1. Research Assistant
Upload research papers and ask questions about methodologies, findings, and conclusions.

### 2. Study Helper
Upload textbooks and lecture notes, then quiz yourself with questions.

### 3. Document Q&A
Upload manuals, guides, or documentation and quickly find specific information.

### 4. Legal Review
Upload contracts and legal documents to extract key information.

### 5. Business Intelligence
Upload reports and analyze business data through natural language queries.

---

## 🔒 Privacy & Security

- ✅ All processing happens **locally**
- ✅ No data sent to external servers (after initial model download)
- ✅ Documents stay on your computer
- ✅ Query history stored locally only

---

## 📞 Support

If you encounter issues:

1. Check this guide's Troubleshooting section
2. Review the error message carefully
3. Check GitHub Issues: https://github.com/Sathvik2005/Knowledge-base-Search-engine-unthinkable-/issues
4. Create a new issue with:
   - Error message
   - Steps to reproduce
   - System information (OS, Python version)

---

## 🎉 You're Ready!

Start by:
1. Running the application
2. Uploading the `sample_document.txt` file
3. Asking: "What is RAG?"
4. Exploring other features

**Enjoy using Answer IQ!** 🚀
