import streamlit as st
import PyPDF2
from huggingface_hub import InferenceClient
import requests
from streamlit.errors import StreamlitSecretNotFoundError

# ---------------------------------------------------------------------
# ✅ STREAMLIT CONFIGURATION
# ---------------------------------------------------------------------
st.set_page_config(page_title="Study Vibes - PDF Summarizer", page_icon="📚", layout="centered")

st.markdown("""
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1503676260728-1c00da094a0b');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.94);
        padding: 2rem;
        border-radius: 15px;
    }
    .stButton button {
        background-color: #5c6bc0;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #3949ab;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# ✅ INITIALIZE HUGGING FACE CLIENT
# ---------------------------------------------------------------------
def init_hf_client():
    try:
        HF_TOKEN = st.secrets["HF_API_TOKEN"]
    except StreamlitSecretNotFoundError:
        st.error("🚨 Missing Hugging Face token in secrets.toml. Add it as:\n\n[general]\nHF_API_TOKEN='your_token_here'")
        st.stop()

    try:
        client = InferenceClient(token=HF_TOKEN)

        # Quick token test
        try:
            client.summarization("Token check", model="sshleifer/distilbart-cnn-12-6")
        except Exception as e:
            if "403" in str(e) or "Forbidden" in str(e):
                st.error(
                    "❌ 403 Forbidden — Your Hugging Face token lacks Inference API access.\n"
                    "Go to https://huggingface.co/settings/tokens → Create token with **Inference API** permission."
                )
                st.stop()
        return client, HF_TOKEN
    except Exception as e:
        st.error(f"❌ Failed to initialize Hugging Face client: {e}")
        st.stop()

client, HF_TOKEN = init_hf_client()
MODEL_NAME = "facebook/bart-large-cnn"

# ---------------------------------------------------------------------
# ✅ HELPER FUNCTIONS
# ---------------------------------------------------------------------
def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def split_into_chunks(text, chunk_size=700):
    """Splits text into chunks for summarization."""
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += para + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = para + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks


def split_text(text, max_len=3000):
    """Split text into smaller overlapping chunks."""
    sentences = text.split(". ")
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 > max_len:
            chunks.append(current.strip())
            current = s + ". "
        else:
            current += s + ". "
    if current:
        chunks.append(current.strip())
    return chunks


# def split_text(text, max_len=3000):
#     """Split text into smaller overlapping chunks."""
#     sentences = text.split(". ")
#     chunks, current = [], ""
#     for s in sentences:
#         if len(current) + len(s) + 1 > max_len:
#             chunks.append(current.strip())
#             current = s + ". "
#         else:
#             current += s + ". "
#     if current:
#         chunks.append(current.strip())
#     return chunks


# def summarize_text(text, model=MODEL_NAME):
#     """Summarizes text using Hugging Face API (with fallback)."""
#     if not text or len(text.strip()) == 0:
#         return None

#     try:
#         res = client.summarization(text, model=model)
#         if isinstance(res, dict):
#             return res.get("summary_text") or res.get("generated_text") or str(res)
#         if hasattr(res, "summary_text"):
#             return res.summary_text
#         return str(res)

#     except Exception as e:
#         msg = str(e)
#         if "403" in msg or "Forbidden" in msg:
#             st.error(
#                 "❌ 403 Forbidden — Token missing Inference API permission.\n"
#                 "➡️ Fix: Create a new Hugging Face token with Inference API access and update secrets.toml."
#             )
#             return None

#         # HTTP fallback (for raw API)
#         try:
#             headers = {"Authorization": f"Bearer {HF_TOKEN}"}
#             # url = f"https://api-inference.huggingface.co/models/{model}"
#             url = f"https://router.huggingface.co/hf-inference/models/{model}"
#             payload = {"inputs": text, "parameters": {"max_new_tokens": 100}}
#             r = requests.post(url, headers=headers, json=payload, timeout=30)
#             if r.status_code == 200:
#                 data = r.json()
#                 if isinstance(data, list) and len(data) and isinstance(data[0], dict):
#                     return data[0].get("summary_text") or data[0].get("generated_text") or str(data)
#                 if isinstance(data, dict):
#                     return data.get("summary_text") or data.get("generated_text") or str(data)
#                 return str(data)
#             else:
#                 st.error(f"⚠️ HTTP fallback failed: {r.status_code} — {r.text}")
#                 return None
#         except Exception as e2:
#             st.error(f"Summarization failed: {e} | fallback error: {e2}")
#             return None



def summarize_text(text, model=MODEL_NAME):
    """Summarizes text safely using Hugging Face API (handles long text + fallback)."""
    if not text or len(text.strip()) == 0:
        return None

    # ✅ Limit text length to avoid token overflow
    text = text[:3500]

    try:
        res = client.summarization(text, model=model)
        if isinstance(res, dict):
            return res.get("summary_text") or res.get("generated_text") or str(res)
        if hasattr(res, "summary_text"):
            return res.summary_text
        return str(res)

    except Exception as e:
        msg = str(e)
        if "403" in msg or "Forbidden" in msg:
            st.error(
                "❌ 403 Forbidden — Token missing Inference API permission.\n"
                "➡️ Create a new token with Inference API access and update secrets.toml."
            )
            return None

        # ✅ Fallback (fixed endpoint + text limit)
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            url = f"https://router.huggingface.co/hf-inference/models/{model}"
            payload = {
                "inputs": text,
                "parameters": {"max_new_tokens": 100, "do_sample": False}
            }
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    return data[0].get("summary_text") or data[0].get("generated_text") or str(data)
                if isinstance(data, dict):
                    return data.get("summary_text") or data.get("generated_text") or str(data)
                return str(data)
            else:
                st.error(f"⚠️ HTTP fallback failed: {r.status_code} — {r.text}")
                return None
        except Exception as e2:
            st.error(f"Summarization failed: {e} | fallback error: {e2}")
            return None


def summarize_chunks(chunks):
    """Summarizes multiple chunks sequentially."""
    summaries = []
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 50:
            continue
        st.info(f"📄 Processing chunk {i + 1}/{len(chunks)}...")
        summary = summarize_text(chunk)
        summaries.append(summary if summary else f"[Chunk {i + 1} failed]")
    return summaries


def create_final_summary(summaries):
    """Combines chunk summaries into one concise version."""
    valid = [s for s in summaries if s and not s.startswith("[Chunk")]
    if not valid:
        return "⚠️ No summaries generated."
    combined = " ".join(valid)
    if len(combined) < 200:
        return combined
    st.info("📝 Creating final summary...")
    final = summarize_text(combined)
    return final if final else combined

# ---------------------------------------------------------------------
# ✅ STREAMLIT UI
# ---------------------------------------------------------------------
st.title("📚 Study Vibes – AI PDF Summarizer")
st.write("Upload a PDF and get an AI-powered summary!")

# Test connection
if st.button("🔍 Test Connection"):
    with st.spinner("Testing Hugging Face API connection..."):
        test = summarize_text("Artificial intelligence is transforming the world through machine learning and deep learning technologies.")
        if test:
            st.success(f"✅ Working! Example summary:\n\n{test}")
        else:
            st.error("❌ Connection failed — check your token or network.")

# PDF uploader
pdf = st.file_uploader("📂 Upload PDF", type=["pdf"])

if pdf:
    with st.spinner("📖 Reading PDF..."):
        text = extract_text_from_pdf(pdf)

    if not text.strip():
        st.error("❌ No text found in the PDF.")
    else:
        chunks = split_into_chunks(text)
        st.success(f"✅ Extracted {len(chunks)} chunks for summarization.")

        with st.expander("👁️ Preview Text"):
            st.text(chunks[0][:300] + "...")

        if st.button("🧠 Generate Summary"):
            summaries = summarize_chunks(chunks)
            if summaries:
                final = create_final_summary(summaries)
                st.subheader("📝 Final Summary")
                st.success(final)

                st.subheader("📋 Chunk Summaries")
                for i, s in enumerate(summaries, 1):
                    st.markdown(f"**{i}.** {s}")
            else:
                st.error("❌ Summarization failed.")
