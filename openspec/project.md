# Project Context: Email Spam Classification

## Project Overview
**Name:** Email Spam Classification System  
**Course:** Machine Learning 2025  
**Type:** Individual Homework (HW3)  
**Weight:** 10%

## Purpose
Build an email spam classification system using machine learning techniques, following the CRISP-DM methodology and OpenSpec workflow for spec-driven development.

## Tech Stack

### Programming Language
- Python 3.8+

### Core Libraries
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn
- **Visualization:** plotly, matplotlib, seaborn
- **Web Framework:** streamlit
- **Text Processing:** nltk (optional)

### Development Tools
- **Version Control:** Git, GitHub
- **Deployment:** Streamlit Cloud
- **Documentation:** Markdown

## Dataset
- **Source:** SMS Spam Collection Dataset
- **File:** `sms_spam_clean.csv`
- **Features:** text messages with spam/ham labels
- **Preprocessing:** cleaning, tokenization, vectorization (TF-IDF)

## ML Pipeline

### 1. Data Preprocessing
- Load and explore dataset
- Clean text data (remove punctuation, lowercase, etc.)
- Handle missing values
- Split into training and testing sets

### 2. Feature Engineering
- TF-IDF vectorization
- Feature selection (top N features)

### 3. Model Training
- Logistic Regression
- Naïve Bayes (MultinomialNB)
- Support Vector Machine (SVM)
- Random Forest (current implementation)

### 4. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve and AUC
- Model comparison

### 5. Deployment
- Streamlit interactive web application
- Real-time prediction interface
- Visualization dashboard

## Conventions

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings for functions
- Keep functions modular and reusable

### Git Workflow
- Commit messages tagged by phase:
  - `Phase 1 – Preprocessing`
  - `Phase 2 – Modeling`
  - `Phase 3 – Evaluation`
  - `Phase 4 – Deployment`
- Keep commits atomic and descriptive

### Documentation
- Clear README with setup instructions
- Inline comments for complex logic
- OpenSpec change proposals for new features

## Project Structure
```
5114056002_HW3/
├── openspec/
│   ├── project.md          # This file
│   ├── AGENTS.md           # Workflow guidelines
│   └── proposals/          # Change proposals
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── sms_spam_clean.csv      # Dataset
├── README.md               # Project documentation
└── .gitignore              # Git ignore rules
```

## Success Criteria
- ✅ Streamlit demo deployed and accessible
- ✅ Multiple ML models implemented and compared
- ✅ Clear visualizations and metrics
- ✅ Complete OpenSpec documentation
- ✅ Clean, well-documented code
