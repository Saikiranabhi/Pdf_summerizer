# 📚 Study Vibes – AI PDF Summarizer

A Streamlit web app that extracts text from PDF files and generates AI-powered summaries using Hugging Face's BART model.

## Features

- 📄 Upload and extract text from PDF files
- 🤖 AI-powered summarization using `facebook/bart-large-cnn`
- 📊 Chunk-based processing for large documents
- 🎨 Clean, modern UI with custom styling
- ✅ Connection testing and error handling

## Setup

1. **Clone the repository**
```bash
git clone https://github.com/Saikiranabhi/Pdf_summerizer.git
cd Pdf_sumerizer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Hugging Face API token**

Create `.streamlit/secrets.toml` and add your token:
```toml
HF_API_TOKEN = "your_huggingface_token_here"
```

Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens) with **Inference API** permissions enabled.

## Usage

Run the app:
```bash
streamlit run app.py
```

1. Click "Test Connection" to verify your API setup
2. Upload a PDF file
3. Click "Generate Summary" to get AI-powered insights

## Requirements

- Python 3.7+
- Streamlit
- PyPDF2
- Hugging Face Hub
- Valid Hugging Face API token

## License

MIT License
