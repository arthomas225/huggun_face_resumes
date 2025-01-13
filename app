import streamlit as st
import os
import PyPDF2
import docx
import re
import nltk
import zipfile
import tempfile
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK data is downloaded
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)
#nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
#nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
#nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
# Remove or comment out the line below
# nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)

model = SentenceTransformer('all-mpnet-base-v2')

# To retain selections across reruns
if 'selected_resumes' not in st.session_state:
    st.session_state.selected_resumes = set()

# Store the actual files in session state as well
if 'all_resumes_data' not in st.session_state:
    st.session_state.all_resumes_data = {}

def extract_text_snippet(content, query_embedding):
    sentences = sent_tokenize(content)
    if not sentences:
        return ""
    candidate_sentences = [s for s in sentences if len(s.strip()) > 3]

    if not candidate_sentences:
        return ""

    preprocessed_sents = [preprocess_text(s) for s in candidate_sentences]
    sent_embeddings = model.encode(preprocessed_sents)
    sims = cosine_similarity([query_embedding], sent_embeddings)[0]
    max_idx = sims.argmax()
    return candidate_sentences[max_idx].strip()

def read_resume(file):
    ext = os.path.splitext(file.name)[1].lower()
    content = ""
    try:
        if ext == ".pdf":
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                content += page.extract_text() + "\n"
        elif ext == ".docx":
            doc = docx.Document(file)
            content = '\n'.join([para.text for para in doc.paragraphs])
        elif ext == ".txt":
            content = file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
    return content

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmas)

def compute_similarity(job_description, resumes, progress_bar):
    preprocessed_jd = preprocess_text(job_description)
    jd_embedding = model.encode(preprocessed_jd)

    resume_results = []
    snippets = []
    for idx, (resume_name, resume_text) in enumerate(resumes.items()):
        processed_resume = preprocess_text(resume_text)
        resume_embedding = model.encode(processed_resume)
        similarity_score = cosine_similarity([jd_embedding], [resume_embedding])[0][0]
        snippet = extract_text_snippet(resume_text, jd_embedding)
        resume_results.append((resume_name, similarity_score))
        snippets.append(snippet)
        progress_bar.progress((idx + 1) / len(resumes))

    sorted_resumes = sorted(zip(resume_results, snippets), key=lambda x: x[0][1], reverse=True)[:25]
    return sorted_resumes

st.title("Resume Search and Ranking Tool")

uploaded_zip = st.file_uploader("Upload a ZIP containing resumes (PDF, DOCX, TXT)", type=['zip'])
job_description = st.text_area("Enter Job Description:")
progress_bar = st.progress(0)

# We put these outside so we can re-display after selection
if uploaded_zip and job_description and st.button("Rank Resumes"):
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Clear old data
        st.session_state.all_resumes_data = {}
        resumes = {}
        
        for root, _, files in os.walk(temp_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                with open(file_path, 'rb') as f:
                    content = read_resume(f)
                    if content:
                        # Store file content in a dict
                        resumes[file_name] = content
                        # Store the original binary data in session_state
                        with open(file_path, 'rb') as original_file:
                            st.session_state.all_resumes_data[file_name] = original_file.read()

        progress_bar.progress(0)
        ranked_resumes = compute_similarity(job_description, resumes, progress_bar)
        progress_bar.progress(1)

        st.success("Resumes ranked successfully! Top 25 shown below.")

        for i, ((resume_name, score), snippet) in enumerate(ranked_resumes):
            st.write(f"{i + 1}. {resume_name} - Similarity: {score:.2f}")
            st.write(f"Snippet: {snippet}")
            if st.checkbox(f"Select {resume_name}", key=f"select_{resume_name}"):
                st.session_state.selected_resumes.add((resume_name, score, snippet))
            else:
                st.session_state.selected_resumes.discard((resume_name, score, snippet))


# --- Show selected resumes (and allow user to download) ---
if st.session_state.selected_resumes:
    st.markdown("### Download Your Selected Resumes")

    # 1) Download the text summary as .txt, as before
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        for i, (resume_name, score, snippet) in enumerate(st.session_state.selected_resumes):
            temp_file.write(f"{i + 1}. {resume_name} - Similarity: {score:.2f}\nSnippet: {snippet}\n\n".encode())
        txt_file_path = temp_file.name

    st.download_button(
        "Download Selected Results (Text Summary)",
        data=open(txt_file_path, "rb").read(),
        file_name="selected_ranked_resumes.txt",
        mime="text/plain"
    )

    # 2) Download the original selected files in one ZIP
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_zip:
        with zipfile.ZipFile(temp_zip.name, 'w') as z:
            for (resume_name, _, _) in st.session_state.selected_resumes:
                # Get the binary data from session state
                file_data = st.session_state.all_resumes_data.get(resume_name)
                if file_data is not None:
                    z.writestr(resume_name, file_data)
        zip_file_path = temp_zip.name

    st.download_button(
        "Download Selected Resume Files (ZIP)",
        data=open(zip_file_path, "rb").read(),
        file_name="selected_resumes.zip",
        mime="application/zip"
    )

else:
    st.warning("Please upload a ZIP file and enter a job description, then click 'Rank Resumes'.")
