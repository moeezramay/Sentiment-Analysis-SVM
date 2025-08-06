# Sentiment Analysis Project

This project analyzes text to determine if the sentiment is positive, negative, or neutral. It uses machine learning to achieve **91% accuracy**.

## What This Project Does

- Analyzes text comments and determines their sentiment (positive/negative/neutral)
- Uses advanced machine learning techniques for high accuracy
- Includes multiple models: SVM, BERT, Naive Bayes, Random Forest, and XGBoost
- Provides easy-to-use notebooks and scripts

## Quick Start

### 1. Get the Project
```bash
git clone <repository-url>
cd Sentiment-Analysis-SVM
```

### 2. Install Requirements
```bash
pip install pandas numpy scikit-learn matplotlib jupyter transformers torch
```

### 3. Run the Analysis
Choose one of these options:

**Option A: Use Jupyter Notebook (Recommended)**
```bash
jupyter notebook
```
Then open any of the `.ipynb` files in the model folders.

**Option B: Run Python Scripts**
```bash
# SVM Model
python SVM_Model/sentiment.py

# BERT Model  
python BERT/simple_bert.py

# Naive Bayes
python NB/simple_naive_bayes.py

# Random Forest
python RandomForestModel/RFP.py

# XGBoost
python XG_Boost/simple_xgboost.py
```

## Project Structure

```
Sentiment-Analysis-SVM/
├── SVM_Model/              # Support Vector Machine implementation
├── BERT/                   # BERT transformer model
├── NB/                     # Naive Bayes classifier
├── RandomForestModel/      # Random Forest algorithm
├── XG_Boost/              # XGBoost gradient boosting
├── data.csv               # Original dataset
└── cleaned_data.csv       # Preprocessed data
```

## Model Performance

| Model | Accuracy | Best For |
|-------|----------|----------|
| SVM | 91.16% | Overall best performance |
| BERT | ~90% | Advanced language understanding |
| Random Forest | ~88% | Good balance of speed/accuracy |
| Naive Bayes | ~85% | Fast and simple |
| XGBoost | ~87% | Gradient boosting approach |

## How It Works

1. **Data Preparation**: Text is cleaned and preprocessed
2. **Feature Extraction**: Words and characters are converted to numerical features
3. **Model Training**: Machine learning models learn patterns from the data
4. **Prediction**: New text is classified as positive, negative, or neutral

## Key Features

- **Multiple Models**: Compare different machine learning approaches
- **High Accuracy**: 91% accuracy with the SVM model
- **Easy to Use**: Simple scripts and notebooks
- **Visual Results**: Charts and graphs showing performance
- **Clean Code**: Well-organized and documented

## Requirements

- Python 3.7+
- pandas, numpy, scikit-learn
- matplotlib, jupyter
- transformers, torch (for BERT)

## Getting Help

If you run into issues:
1. Check that all requirements are installed
2. Make sure you're in the correct directory
3. Try running the Jupyter notebooks first

## Contributing

Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Improve documentation

---

*This project demonstrates various machine learning approaches for sentiment analysis, from simple classifiers to advanced transformer models.*
