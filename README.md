# Sentiment Analysis Tool

A machine learning-based text sentiment analysis tool that classifies comments as positive, negative, or neutral with **91.16% accuracy**.

## 🎯 Features

- **High Accuracy**: Achieved 91.16% classification accuracy using SVM with optimized feature engineering
- **Text Preprocessing**: Handles cleaned text data with special characters removed
- **Advanced Feature Engineering**: Combines word-level and character-level TF-IDF features
- **Class Balance Handling**: Optimized for imbalanced sentiment distributions
- **Fast Training**: Efficient SVM implementation without unnecessary complexity

## 📊 Performance Evolution

| Model/Improvement | Accuracy | Change |
|------------------|----------|---------|
| Initial Logistic Regression | 86.73% | Baseline |
| Basic SVM (LinearSVC) | 89.2% | +2.47% |
| Enhanced TF-IDF Features | 90.70% | +1.5% |
| **Final Optimized SVM** | **91.16%** | **+4.43%** |

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Sentimental-analysis
   ```

2. **Install required packages:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib jupyter
   ```

## 📁 Project Structure

```
Sentimental-analysis/
├── sentiment.ipynb          # Main Jupyter notebook with complete pipeline
├── cleaned_data.csv         # Preprocessed training data
├── data.csv                 # Original dataset
├── sentiment_analysis_results.png  # Visualization results
└── svm_accuracy_results.png        # SVM performance charts
```

## 🚀 Usage

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook sentiment.ipynb
```

### Option 2: Python Script
```bash
python sentiment.py
```

## 🔧 Technical Stack

- **Python** with scikit-learn
- **SVM (LinearSVC)** for classification
- **TF-IDF Vectorization** with feature union
- **Jupyter Notebook** for interactive development
- **Pandas** for data manipulation

## 📈 Model Details

### Feature Engineering
- **Word-level TF-IDF**: 12,000 features with 1-3 n-grams
- **Character-level TF-IDF**: 5,000 features with 3-5 n-grams
- **Feature Union**: Combines both approaches for better representation

### Model Parameters
- **Algorithm**: LinearSVC (Support Vector Machine)
- **Regularization**: C=2.5 (balanced strictness)
- **Max Iterations**: 5,000 for convergence
- **Loss Function**: Hinge loss

## 📊 Results

The model achieves excellent performance across all sentiment classes:

- **Overall Accuracy**: 91.16%
- **Precision**: High across positive, negative, and neutral classes
- **Recall**: Balanced performance for all sentiment types
- **F1-Score**: Strong harmonic mean of precision and recall

## 🎓 Learning Outcomes

This project demonstrates:
- Complete ML pipeline from data preprocessing to model evaluation
- Feature engineering techniques for text classification
- Model selection and hyperparameter tuning
- Performance optimization strategies
- Interactive development with Jupyter notebooks

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

*Built as a learning project to understand machine learning pipelines and text classification techniques.*
