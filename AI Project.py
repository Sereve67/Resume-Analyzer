
import pandas as pd
import nltk
import spacy
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# 🔹 Fix Missing NLTK Data Issues
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english")) - {"java", "python", "sql", "developer", "engineer", "degree", "bachelor"}

# 🔹 Use Medium Model for Better Recognition
spacy.cli.download("en_core_web_md")
nlp = spacy.load("en_core_web_md")

# 🔹 Load Data
df = pd.read_csv("Resume.csv")
df.columns = df.columns.str.strip()

if "Resume_str" not in df.columns:
    raise KeyError("Column 'Resume_str' not found in Resume.csv.")

# 🔹 Text Cleaning Function
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

df["cleaned_resume"] = df["Resume_str"].apply(clean_text)

# 🔹 Predefined Skills List for Matching
skills_list = ["python", "java", "sql", "c++", "machine learning", "deep learning",
               "tensorflow", "pandas", "scikit-learn", "nlp", "data science",
               "excel", "power bi", "tableau", "flask", "django", "aws", "cloud computing"]

# 🔹 Enhanced Entity Recognition Function
def extract_entities(text):
    doc = nlp(text)
    entities = {"Name": "Not Found", "Skills": "Not Found", "Experience": "Not Found", "Education": "Not Found", "Contact": "Not Found"}

    # 🔹 Extract Name
    for ent in doc.ents:
        entity_text = ent.text.strip()
        if ent.label_ == "PERSON" and len(entity_text.split()) > 1:
            entities["Name"] = entity_text.title()

    # 🔹 Extract Education (Looks for 'University' or Degree Keywords)
    for ent in doc.ents:
        entity_text = ent.text.strip().lower()
        if ent.label_ in ["ORG", "GPE"] and ("university" in entity_text or "college" in entity_text):
            entities["Education"] = entity_text.title()

    # 🔹 Extract Experience (Using Regex for Years of Experience)
    experience_match = re.search(r"(\d+)\s*(years|year)\s*experience", text.lower())
    if experience_match:
        entities["Experience"] = experience_match.group(0)

    # 🔹 Extract Contact Information (Phone Number & Email)
    phone_match = re.search(r"\b\d{10}\b", text)  # Matches a 10-digit phone number
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)  # Matches an email

    if phone_match:
        entities["Contact"] = phone_match.group(0)
    if email_match:
        entities["Contact"] = email_match.group(0)

    # 🔹 Extract Skills (Matching Against Predefined Skills List)
    found_skills = [skill for skill in skills_list if skill in text.lower()]
    if found_skills:
        entities["Skills"] = ", ".join(found_skills).title()

    return entities

# 🔹 Apply Entity Extraction
df["Extracted_Entities"] = df["cleaned_resume"].apply(extract_entities)
print(df[["Extracted_Entities"]].head())

# 🔹 Fix Missing Category Column
if "Category" not in df.columns:
    raise KeyError("Column 'Category' not found in Resume.csv.")

# 🔹 TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)  
X = vectorizer.fit_transform(df["cleaned_resume"])
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train Model
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)  
model.fit(X_train, y_train)

# 🔹 Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
