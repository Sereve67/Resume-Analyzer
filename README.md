# AI-Powered Resume Analyzer

## 📌 Project Overview
The **AI-Powered Resume Analyzer** is a machine learning-based web application that automates resume screening. It extracts key details from resumes and predicts the job category of candidates using **Natural Language Processing (NLP) and Machine Learning (ML)**.

## 🚀 Features
- **Upload resumes (CSV or TXT)**
- **Extracts key details:** Name, Skills, Experience, Education, and Contact
- **Predicts job category** using ML
- **Displays model accuracy**
- **Interactive Streamlit-based web interface**

## 🛠️ Technologies Used
- **Python** - Core programming language
- **Pandas** - Data handling
- **spaCy** - Named Entity Recognition (NER)
- **NLTK** - Text preprocessing (stopword removal, tokenization)
- **Regex** - Extracting phone numbers, emails, experience years
- **Scikit-Learn** - Machine Learning (Random Forest Classifier)
- **TF-IDF Vectorization** - Feature extraction
- **Streamlit** - Web interface

## 🔹 Dataset Used
This project uses the **Resume.csv** dataset from **Kaggle** to train and evaluate the model.

## 🔹 How It Works
1. **Data Processing:** Loads and cleans resume text from CSV.
2. **NER Extraction:** Uses **spaCy** to detect key details like Name, Experience, and Education.
3. **ML Model:** Uses **TF-IDF** and **Random Forest Classifier** to predict the job category.
4. **Frontend Interface:** Users can upload resumes via **Streamlit**, view extracted details, and see job predictions.

## 📂 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/AI-Resume-Analyzer.git
   cd AI-Resume-Analyzer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run AI_Project_Frontend.py
   ```

## 📊 Example Output
```
📄 Uploaded Resume: Resume.csv
🔍 Extracted Information:
{
  "Name": "John Doe",
  "Skills": "Python, Machine Learning",
  "Experience": "5 years experience",
  "Education": "MIT",
  "Contact": "Not Found"
}
🏷️ Predicted Job Category: Data Scientist
📈 Model Accuracy: 92.5%
```

## 🎯 Future Enhancements
- Improve **NER accuracy** with a custom-trained model
- Support **PDF and DOCX** file formats
- Enhance **job prediction with deep learning**

## 🤝 Contributing
Pull requests are welcome! Please follow best practices and submit issues for any bugs or improvements.

## 📜 License
This project is licensed under the MIT License.
