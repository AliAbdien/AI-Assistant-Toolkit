# AI Assistant Toolkit

A smart and versatile AI assistant web application built with Streamlit.  
It allows users to interact with PDF documents, web pages, images, YouTube videos, and perform web search and math problem-solving using powerful LLMs.

## Features

- 📄 Chat with PDFs
- 🔗 Analyze and ask questions about website content
- 🖼️ Extract text and generate captions from images
- 🎥 Summarize and interact with YouTube transcripts
- 🔍 Web search assistant using Wikipedia, Arxiv, and DuckDuckGo
- 🧠 Solve math and logic problems step-by-step
- 🧹 Session-based chat history management

## Technologies Used

- Python 3.10+
- Streamlit
- LangChain
- HuggingFace Transformers
- FAISS (for vector search)
- Groq API (for LLMs)
- YouTube Transcript API
- OCR (Tesseract)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/ai-assistant-toolkit.git
cd ai-assistant-toolkit
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your Groq API Key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

## Notes

- Make sure you have Tesseract installed for OCR to work.
- The app supports multiple LLM models from Groq, including Llama, Gemma, Compound, and others.

---

## License

This project is licensed under the MIT License.

---
