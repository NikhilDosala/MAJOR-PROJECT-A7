import streamlit as st import pandas as pd import os
import fitz import nltk import torch
import matplotlib.pyplot as plt import seaborn as sns
import re
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util from rouge_score import rouge_scorer
from fpdf import FPDF
import hashlib # ✅ for password hashing


# =============================
# □ App Configuration
# =============================
st.set_page_config(page_title="Legal & Financial Document Summarizer", layout="wide") nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# =============================
# 📂 Setup Paths
# =============================
BASE_DIR = os.path.dirname(os.path.abspath( file )) USERS_FILE = os.path.join(BASE_DIR, "users.csv")

# ============================= # =============================
 


def hash_password(password):
"""Encrypt password using SHA256 (secure & simple).""" return hashlib.sha256(password.encode()).hexdigest()

def verify_password(input_password, stored_password_hash): """Check if user-entered password matches stored hash."""
return hash_password(input_password) == stored_password_hash

# Create users.csv if not exists
if not os.path.exists(USERS_FILE):
df = pd.DataFrame(columns=["username", "password"]) df.to_csv(USERS_FILE, index=False)
st.info(f"🆕 Creating user database at: {USERS_FILE}")


def user_exists(username):
users = pd.read_csv(USERS_FILE)
return username in users["username"].values


def add_user(username, password): users = pd.read_csv(USERS_FILE)
new_user = pd.DataFrame([[username, hash_password(password)]], columns=["username", "password"])
users = pd.concat([users, new_user], ignore_index=True) users.to_csv(USERS_FILE, index=False)

def verify_user(username, password): users = pd.read_csv(USERS_FILE)
if username in users["username"].values:
stored_hash = users.loc[users["username"] == username, "password"].values[0] return verify_password(password, stored_hash)
return False
 


# ============================= # 🔐 LOGIN / SIGNUP SYSTEM
# =============================
if "logged_in" not in st.session_state: st.session_state.logged_in = False

menu = ["Login", "Sign Up"]
choice = st.sidebar.selectbox("Navigation", menu)


if not st.session_state.logged_in: if choice == "Login":
st.title("🔐 Login to Legal & Financial Document Summarizer")


username = st.text_input("👤 Username")
password = st.text_input("🔑 Password", type="password")


if st.button("Login"):
if verify_user(username, password): st.session_state.logged_in = True st.session_state.username = username st.success(f"✅ Welcome, {username}!") st.rerun()
else:
st.error("❌ Invalid username or password")


elif choice == "Sign Up":
st.title("🆕 Create a New Account")


new_username = st.text_input("👤 Choose a Username")
new_password = st.text_input("🔑 Choose a Password", type="password") confirm_password = st.text_input("🔁 Confirm Password", type="password")
 


if st.button("Create Account"):
if not new_username or not new_password: st.warning("⚠️ Please fill in all fields.")
elif new_password != confirm_password:
st.error("❌ Passwords do not match.") elif user_exists(new_username):
st.error("⚠️ Username already exists. Please choose another.") else:
add_user(new_username, new_password)
st.success("✅ Account created successfully! Please login from the sidebar.") st.stop()

# ============================= # 🏠 MAIN DASHBOARD
# =============================
st.sidebar.title("📘 Legal & Financial Document Summarizer") st.sidebar.write(f"👋 Logged in as: `{st.session_state.username}`")

if st.sidebar.button("🚪 Logout"): st.session_state.logged_in = False st.rerun()

st.title("📄 Legal & Financial Document Summarizer Dashboard")
st.write("Upload a **Legal or Financial PDF**, and this system will generate both Extractive and AI-based summaries, along with insights and downloadable reports.")

# =============================
# 📂 File Upload
# =============================
uploaded_file = st.file_uploader("📤 Upload your Legal or Financial PDF file", type=["pdf"
 


if uploaded_file is not None:
pdf_reader = fitz.open(stream=uploaded_file.read(), filetype="pdf") text = ""
for page in pdf_reader:
text += page.get_text("text")


if len(text.strip()) < 100:
st.error("❌ PDF text too short or empty. Please upload a proper document.") else:
st.success("✅ PDF uploaded and text extracted successfully!") sentences = sent_tokenize(text)
st.write(f"📄 Total sentences detected: **{len(sentences)}**")


# =============================
# □ Extractive Summarization
# =============================
st.subheader("🔹 Extracting Main Points (Sentence Ranking)")


model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences, convert_to_tensor=True) central_vector = torch.mean(embeddings, dim=0)
similarities = util.cos_sim(embeddings, central_vector) ranked_indices = similarities.squeeze().argsort(descending=True) ranked_sentences = [sentences[i] for i in ranked_indices]

top_n = 7 if len(sentences) > 7 else len(sentences)

def clean_sentences(sentences): filtered = []
for s in sentences: s = s.strip()
if len(s) < 15 or "	" in s:
 


continue

if re.match(r"^\d+(\.\d+)?$", s.strip()): continue
if s.lower().startswith(("between", "this deed", "executed at", "of the one part")): continue
filtered.append(s) return filtered

cleaned_summary = clean_sentences(ranked_sentences[:top_n]) extractive_summary = " ".join(cleaned_summary)

st.markdown("### □ Extractive Summary (Key Legal Points)") for i, s in enumerate(cleaned_summary, 1):
st.markdown(f"**{i}.** {s.strip()}")


# =============================
# 🤖 Abstractive Summarization
# =============================
st.subheader("🤖 AI Generated Summary (Abstractive)")
# Using text2text-generation for compatibility with newer transformers from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tokenize and generate summary
inputs = tokenizer([text[:1024]], max_length=1024, return_tensors="pt", truncation=True)
summary_ids = model.generate(inputs["input_ids"], max_length=180, min_length=60, do_sample=False)
 


ai_summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
st.write(ai_summary)

# =============================
# 📊 Visualization
# =============================
st.subheader("📊 Visualization and Analytics")


# Sentence Importance
st.write("### Sentence Importance (Extractive Ranking)") fig1, ax1 = plt.subplots()
ax1.bar(range(len(similarities.squeeze())), similarities.squeeze().tolist(), color='skyblue') ax1.set_xlabel("Sentence Index")
ax1.set_ylabel("Importance Score") ax1.set_title("Sentence Importance Ranking") st.pyplot(fig1)

# Compression Comparison
st.write("### Text Compression Comparison") orig_len = len(text.split())
ext_len = len(extractive_summary.split()) abs_len = len(ai_summary.split())
fig2, ax2 = plt.subplots()
ax2.bar(['Original', 'Extractive', 'Abstractive'], [orig_len, ext_len, abs_len], color=['orange', 'skyblue', 'lightgreen'])
ax2.set_ylabel("Word Count") ax2.set_title("Compression Comparison") st.pyplot(fig2)

# ROUGE Metrics
 


st.write("### ROUGE Accuracy Scores")
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True) scores = scorer.score(extractive_summary, ai_summary)
rouge1 = scores['rouge1'].fmeasure

fig3, ax3 = plt.subplots()
ax3.bar(['ROUGE-1', 'ROUGE-L'], [rouge1, rougeL], color='purple') ax3.set_ylim(0, 1)
ax3.set_ylabel("Score (0-1)") ax3.set_title("ROUGE Evaluation Metrics") st.pyplot(fig3)

st.success("✅ Analysis Completed Successfully!")


# =============================
# 📥 PDF Download
# ============================= # =============================
# 📥 PDF Download (Unicode Safe)
# ============================= # =============================
# 📥 PDF Download (Unicode Safe)
# =============================


if 'extractive_summary' in locals() and 'ai_summary' in locals(): def clean_text(txt):
"""Remove or replace non-ASCII and special characters safely.""" replacements = {
'’': "'", '‘': "'", '“': '"', '”': '"',
'–': '-', '—': '-', '•': '*', '\u2022': '*',
'\xa0': ' ', '\n\n': '\n'
}
 


for k, v in replacements.items(): txt = txt.replace(k, v)
txt = re.sub(r'[^\x00-\x7F]+', ' ', txt) # Remove any remaining non-ASCII chars return txt.strip()

extractive_summary_clean = clean_text(extractive_summary) ai_summary_clean = clean_text(ai_summary)

pdf = FPDF() pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Legal & Financial Document Summarizer Report", ln=True, align="C") pdf.ln(10)

pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "Extractive Summary:", ln=True) pdf.set_font("Arial", "", 11)
pdf.multi_cell(0, 8, extractive_summary_clean) pdf.ln(5)

pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "Abstractive Summary:", ln=True) pdf.set_font("Arial", "", 11)
pdf.multi_cell(0, 8, ai_summary_clean) pdf.ln(5)

pdf.set_font("Arial", "I", 10)
pdf.multi_cell(0, 8, f"Generated by: {st.session_state.username}\nUsing Sentence-BERT and BART Transformer models.")
 


# ✅ Safe encoding to avoid Unicode errors
pdf_bytes = bytes(pdf.output(dest='S').encode('latin-1', 'ignore'))

st.download_button(
"⬇️ Download Report (PDF)", data=pdf_bytes, file_name="Legal_Summary_Report.pdf", mime="application/pdf"

)
else:
st.warning("⚠️ Please upload and summarize a document first before downloading the report.")
