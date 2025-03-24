import streamlit as st
import pandas as pd
import nltk
import spacy
import re
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# 🔹 Download NLTK Data (if missing)
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english")) - {"java", "python", "sql", "developer", "engineer", "degree", "bachelor"}

# 🔹 Load Spacy Model
nlp = spacy.load("en_core_web_md")

# 🔹 Skills List for Extraction
skills_list = ["python", "java", "sql", "c++", "machine learning", "deep learning",
               "tensorflow", "pandas", "scikit-learn", "nlp", "data science",
               "excel", "power bi", "tableau", "flask", "django", "aws", "cloud computing"]

# 🔹 Resume Text Cleaning Function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# 🔹 Extract Entities (Name, Skills, Experience, Education, Contact)
def extract_entities(text):
    nlp.max_length = 2_000_000  # Increase to 2 million characters
    doc = nlp(text)
    entities = {"Name": "Not Found", "Skills": "Not Found", "Experience": "Not Found", "Education": "Not Found", "Contact": "Not Found"}

    for ent in doc.ents:
        entity_text = ent.text.strip()
        if ent.label_ == "PERSON" and len(entity_text.split()) > 1:
            entities["Name"] = entity_text.title()
        if ent.label_ in ["ORG", "GPE"] and ("university" in entity_text.lower() or "college" in entity_text.lower()):
            entities["Education"] = entity_text.title()

    experience_match = re.search(r"(\d+)\s*(years|year)\s*experience", text.lower())
    if experience_match:
        entities["Experience"] = experience_match.group(0)

    phone_match = re.search(r"\b\d{10}\b", text)
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)

    if phone_match:
        entities["Contact"] = phone_match.group(0)
    if email_match:
        entities["Contact"] = email_match.group(0)

    found_skills = [skill for skill in skills_list if skill in text.lower()]
    if found_skills:
        entities["Skills"] = ", ".join(found_skills).title()

    return entities

# 🔹 Load Model and Vectorizer
@st.cache_resource
def load_model():
    df = pd.read_csv("Resume.csv")
    df.columns = df.columns.str.strip()

    if "Resume_str" not in df.columns or "Category" not in df.columns:
        st.error("Error: Required columns 'Resume_str' and 'Category' not found in Resume.csv.")
        return None, None

    df["cleaned_resume"] = df["Resume_str"].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["cleaned_resume"])
    y = df["Category"]

    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)

    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(model, "model.pkl")

    return vectorizer, model

vectorizer, model = load_model()

# 🔹 Streamlit UI
st.title("📄 AI-Powered Resume Analyzer")
st.write("Upload a resume to extract details and predict the job category.")

uploaded_file = st.file_uploader("Upload Resume (TXT or CSV)", type=["txt", "csv"])

if uploaded_file is not None:
    try:
        resume_text = uploaded_file.read().decode("utf-8")[:500000]  # Process only the first 500,000 characters
        cleaned_text = clean_text(resume_text)
        extracted_data = extract_entities(resume_text)

        st.subheader("🔍 Extracted Information")
        st.json(extracted_data)

        if vectorizer and model:
            input_features = vectorizer.transform([cleaned_text])
            predicted_category = model.predict(input_features)[0]
            st.success(f"🏷️ **Predicted Job Category:** {predicted_category}")
        else:
            st.error("Model could not be loaded. Check if Resume.csv is correct.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
