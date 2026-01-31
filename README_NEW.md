# 🧠 Answer IQ - RAG Knowledge Base Search Engine

An advanced **Retrieval-Augmented Generation (RAG)** system with voice support, powered by LangChain, Whisper, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## 🌟 Features

- **📚 Multi-Document Processing**: Upload and process multiple PDF and text documents
- **🧠 Semantic Search**: Advanced vector-based retrieval using FAISS
- **🤖 AI-Powered Answers**: Context-aware responses using transformer models
- **🎤 Voice Integration**: OpenAI Whisper-powered audio transcription
- **📊 Query History**: Automatic tracking of all searches and answers
- **⚡ Real-time Processing**: Fast document embedding and retrieval
- **🎨 Modern UI**: Beautiful, intuitive Streamlit interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster processing

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Sathvik2005/Knowledge-base-Search-engine-unthinkable-
cd Knowledge-base-Search-engine-unthinkable-
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Access the app**
Open your browser and navigate to `http://localhost:8501`

## 📖 Usage Guide

### 1. Upload Documents

- Click on the sidebar **"Upload Documents"** section
- Select PDF or TXT files from your computer
- Click **"Build Knowledge Base"** to process the documents

### 2. Ask Questions (Text)

- Navigate to the **"💬 Text Query"** tab
- Type your question in the text area
- Optional: Enable "Concise Mode" for brief answers
- Click **"🔍 Search"** to get your answer

### 3. Ask Questions (Voice)

- Navigate to the **"🎤 Voice Query"** tab
- Upload an audio file (WAV, MP3, M4A, OGG)
- Click **"🎧 Transcribe & Search"**
- The system will transcribe your audio and search the knowledge base

### 4. View History

- Navigate to the **"📜 History"** tab
- View all previous queries and answers
- Click **"🔄 Refresh History"** to update

## 🔧 Technology Stack

- **[LangChain](https://python.langchain.com/)**: Document processing and RAG pipeline
- **[HuggingFace Transformers](https://huggingface.co/transformers/)**: Language models and embeddings
- **[OpenAI Whisper](https://github.com/openai/whisper)**: Speech-to-text transcription
- **[FAISS](https://github.com/facebookresearch/faiss)**: Vector similarity search
- **[Streamlit](https://streamlit.io/)**: Interactive web interface
- **[PyTorch](https://pytorch.org/)**: Deep learning framework

## 🤖 Models Used

| Component | Model | Description |
|-----------|-------|-------------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Fast, efficient sentence embeddings |
| LLM | `microsoft/DialoGPT-medium` | Conversational text generation |
| Voice | `OpenAI Whisper (base)` | Speech-to-text transcription |

## 📁 Project Structure

```
Knowledge-base-Search-engine-unthinkable-/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── rag_cache/                      # Cache directory (auto-created)
├── query_history.json              # Query history (auto-created)
│
└── notebooks/                      # Jupyter notebooks (legacy)
    ├── knoledge_bases_searchbengine (1).ipynb
    ├── knoledge_bases_searchbengine (2).ipynb
    └── unthinkable_knowledge_base_search_engine_.ipynb
```

## ⚙️ Configuration

Edit the `CONFIG` dictionary in `app.py` to customize:

```python
CONFIG = {
    "chunk_size": 500,              # Text chunk size for processing
    "chunk_overlap": 100,            # Overlap between chunks
    "embedding_model": "...",        # HuggingFace embedding model
    "llm_model": "...",              # HuggingFace LLM model
    "retrieval_k": 5,                # Number of documents to retrieve
    "max_tokens": 300,               # Max tokens for generation
    "whisper_model": "base",         # Whisper model size
}
```

## 🎯 Use Cases

- **Research**: Query academic papers and documents
- **Documentation**: Search technical documentation
- **Legal**: Analyze contracts and legal documents
- **Education**: Study from textbooks and notes
- **Business**: Extract insights from reports
- **Personal**: Organize and search personal documents

## 🛠️ Troubleshooting

### Issue: Out of Memory Error

**Solution**: Reduce `chunk_size` and `retrieval_k` in the configuration, or use a smaller LLM model.

### Issue: Slow Processing

**Solution**: 
- Ensure you have a GPU available
- Use smaller models (e.g., `gpt2` instead of `DialoGPT-medium`)
- Reduce the number of uploaded documents

### Issue: Whisper Not Working

**Solution**: 
- Install ffmpeg: `sudo apt-get install ffmpeg` (Linux) or `brew install ffmpeg` (Mac)
- Ensure audio files are in supported formats

## 📝 Development

### Running in Development Mode

```bash
streamlit run app.py --server.runOnSave true
```

### Testing

```bash
python -m pytest tests/
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for the Whisper model
- HuggingFace for transformers and model hosting
- LangChain for the RAG framework
- Facebook Research for FAISS
- Streamlit for the amazing UI framework

## 📧 Contact

**Developer**: Sathvik  
**GitHub**: [@Sathvik2005](https://github.com/Sathvik2005)

---

⭐ If you find this project useful, please consider giving it a star!

## 🔮 Future Enhancements

- [ ] Support for more document formats (DOCX, HTML, etc.)
- [ ] Multi-language support
- [ ] Custom model fine-tuning
- [ ] Cloud deployment options
- [ ] API endpoints
- [ ] Batch processing
- [ ] Advanced filtering and search options
- [ ] Export functionality

---

**Made with ❤️ and AI**
