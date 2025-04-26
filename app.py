import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pytesseract
import torch
import tempfile
import json
from youtube_transcript_api import YouTubeTranscriptApi
import requests

# Load environment variables
load_dotenv()

# Get API keys from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="AI Assistant Toolkit", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "reset_session" not in st.session_state:
    st.session_state.reset_session = False

# === FORMAT RESPONSE ===
def format_llm_output(output):
    if hasattr(output, 'content'):
        return output.content
    elif isinstance(output, dict):
        if 'content' in output:
            return output['content']
        return json.dumps(output, indent=2)
    elif isinstance(output, str):
        if output.startswith("content="):
            return output.replace("content=\"", "", 1).rstrip("\"")
        return output
    return str(output)

# Sidebar for model selection and reset toggle
with st.sidebar:
    st.title("🧠 AI Toolkit")
    selected_model = st.selectbox("Choose LLM Model", [
        "Gemma2-9b-It", "Compound-Beta", "Compound-Beta-Mini", "Llama3-8b-8192", "Llama3-70b-8192",
        "Llama-3.1-8b-Instant", "Llama-3.3-70b-Versatile", "Meta-Llama/Llama-4-Scout-17b-16e-Instruct",
        "Meta-Llama/Llama-4-Maverick-17b-128e-Instruct", "Deepseek-R1-Distill-Llama-70b",
        "Mistral-Saba-24b", "Qwen-Qwq-32b"
    ], index=2)
    st.session_state.reset_session = st.checkbox("🔁 Reset conversation on model change", value=False)

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=selected_model)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Upload buttons styled in columns
col1, col2, col3 = st.columns(3)
pdf_file = None
image_file = None
url_input = ""

with col1:
    pdf_file = st.file_uploader("📄 Upload PDF", type="pdf", label_visibility="collapsed")
with col2:
    image_file = st.file_uploader("🖼️ Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
with col3:
    url_input = st.text_input("🔗 Paste URL (Website or YouTube)", label_visibility="collapsed")

st.markdown("---")

# Text input area
user_input = st.text_input("💬 Ask something:")

# File/URL handling
context = ""
if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        loader = PyPDFLoader(tmp.name)
        docs = loader.load()
        context = "\n".join(doc.page_content for doc in docs)

elif image_file:
    image = Image.open(image_file).convert("RGB")
    processor, model = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"), BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    ocr_text = pytesseract.image_to_string(image)
    context = f"Caption: {caption}\nOCR: {ocr_text}"

elif url_input:
    if "youtube.com" in url_input:
        try:
            video_id = url_input.split("v=")[-1]
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "ar"])
            context = " ".join([x['text'] for x in transcript])
        except:
            st.error("Transcript not available.")
    else:
        full_context = []

        # Load page content
        loader = UnstructuredURLLoader(urls=[url_input], ssl_verify=False)
        docs = loader.load()
        full_context.append("\n".join(doc.page_content for doc in docs))

        # Wikipedia Search
        wiki_response = requests.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "list": "search", "srsearch": url_input, "format": "json"
        })
        if wiki_response.status_code == 200:
            results = wiki_response.json().get("query", {}).get("search", [])
            for item in results[:3]:
                full_context.append(f"Wikipedia Result: {item.get('title')}\n{item.get('snippet')}")

        # Arxiv Search
        arxiv_response = requests.get(f"http://export.arxiv.org/api/query?search_query=all:{url_input}&start=0&max_results=2")
        if arxiv_response.status_code == 200:
            full_context.append("Arxiv Result:\n" + arxiv_response.text)

        # DuckDuckGo Search
        duckduck_response = requests.get(f"https://api.duckduckgo.com/?q={url_input}&format=json")
        if duckduck_response.status_code == 200:
            abstract = duckduck_response.json().get("Abstract", "")
            if abstract:
                full_context.append("DuckDuckGo Abstract:\n" + abstract)

        context = "\n\n".join(full_context)

# Chat interaction
if user_input:
    full_prompt = f"""
    Use the following context to answer:
    {context}

    Question: {user_input}
    """
    answer = llm.invoke(full_prompt)
    formatted_answer = format_llm_output(answer)
    st.session_state.chat_history.append((user_input, formatted_answer))

# Display conversation
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)
