# Sentiment Analysis Tool

A comprehensive machine learning-based text sentiment analysis tool that demonstrates progression from traditional ML to deep learning. Starting with SVM achieving **91.22% accuracy**, then advancing to BERT achieving **97.50% accuracy** for state-of-the-art performance.

## Features

- **Traditional ML Excellence**: Achieved 91.22% classification accuracy using SVM with advanced feature engineering
- **Deep Learning Advancement**: Pushed to 97.50% accuracy using custom BERT training
- **Multiple Model Comparison**: Tested SVM, Random Forest, XGBoost, Naive Bayes, and BERT
- **Custom BERT Implementation**: Built and trained BERT model from scratch
- **Text Preprocessing**: Handles cleaned text data with special characters removed
- **Advanced Feature Engineering**: Combines word-level and character-level TF-IDF features
- **GPU Acceleration**: Optimized BERT training with CUDA support
- **Efficient Training**: BERT achieved 97.50% accuracy in just 1.5 hours
- **Cloud Deployment Ready**: Designed for AWS EC2 server deployment pipeline

## Performance Evolution

| Model/Improvement | Accuracy | Change |
|------------------|----------|---------|
| Initial Logistic Regression | 86.73% | Baseline |
| Basic SVM (LinearSVC) | 89.2% | +2.47% |
| Enhanced TF-IDF Features | 90.70% | +1.5% |
| **Final Optimized SVM** | **91.22%** | **+4.49%** |
| **Custom BERT Model** | **97.50%** | **+10.77%** |

## Model Comparison

| Model | Accuracy | Pros | Cons |
|-------|----------|------|------|
| **BERT (Custom Training)** | **97.50%** | State-of-the-art performance, understands context, excellent for text | Requires GPU, longer training time |
| **SVM (LinearSVC)** | **91.22%** | Excellent for text data, handles high-dimensional features well | Requires feature scaling, slower training |
| **Random Forest** | **82.00%** | Good interpretability, handles non-linear patterns | Struggles with sparse text data, lower accuracy |
| **XGBoost** | **80.62%** | Fast training, good for structured data | Not optimal for sparse text features |
| **Naive Bayes** | **70.78%** | Simple, fast, good baseline | Limited by independence assumption |

### Why BERT Outperforms Traditional Methods:
- **Context Understanding**: BERT captures complex language patterns and context
- **Pre-trained Knowledge**: Leverages vast pre-trained language understanding
- **Attention Mechanism**: Focuses on relevant parts of text for classification
- **Deep Learning**: Can learn sophisticated text representations

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Sentimental-analysis
   ```

2. **Install required packages:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib jupyter transformers torch
   ```

## Project Structure

```
Sentimental-analysis/
├── sentiment.ipynb                    # Main Jupyter notebook with SVM pipeline
├── Sentiment-Analysis-SVM/
│   └── BERT/
│       ├── simple_bert.py            # Custom BERT training implementation
│       └── evaluate_bert.py          # BERT evaluation script
├── RandomForestModel/                 # Random Forest implementation
│   ├── RFMJ.ipynb                    # Random Forest Jupyter notebook
│   └── random_forest_results.png
├── NB/                               # Naive Bayes implementation
├── XG_Boost/                         # XGBoost implementation
├── cleaned_data.csv                  # Preprocessed training data
├── data.csv                          # Original dataset
├── sentiment_analysis_results.png    # Visualization results
└── svm_accuracy_results.png          # SVM performance charts
```

## Usage

### Option 1: Live API (Production - 97.50% Accuracy)
```bash
# Test the live API
curl -X POST http://16.16.187.62:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'

# Health check
curl http://16.16.187.62:5000/health
```

### Option 2: SVM Model (Traditional ML - 91.22% Accuracy)
```bash
jupyter notebook sentiment.ipynb
```

### Option 3: BERT Model (Deep Learning - 97.50% Accuracy)
```bash
cd Sentiment-Analysis-SVM/BERT
python simple_bert.py
```
**Note:** BERT model checkpoints are not included due to size (>100MB). The script will train the model from scratch (~1.5 hours) or download pre-trained model automatically.

### Option 4: Random Forest Model (For Comparison)
```bash
jupyter notebook RandomForestModel/RFMJ.ipynb
```

### Option 5: Python Script
```bash
python sentiment.py
```

## Technical Stack

- **Python** with scikit-learn for traditional ML
- **PyTorch & Transformers** for deep learning
- **SVM (LinearSVC)** for traditional classification
- **BERT (Custom Training)** for state-of-the-art performance
- **TF-IDF Vectorization** with feature union
- **Jupyter Notebook** for interactive development
- **Pandas** for data manipulation
- **CUDA** for GPU acceleration
- **AWS EC2** for cloud deployment (production)
- **Flask** for API server
- **Flask-Limiter** for rate limiting
- **CORS** for frontend integration

## Model Details

### SVM Implementation (91.22% Accuracy)
- **Algorithm**: LinearSVC (Support Vector Machine)
- **Feature Engineering**: Word-level TF-IDF (10,000 features) + Character-level TF-IDF (5,000 features)
- **Regularization**: C=2.5 (balanced strictness)
- **Max Iterations**: 5,000 for convergence
- **Loss Function**: Hinge loss

### BERT Implementation (97.50% Accuracy)
- **Model**: bert-base-uncased (110M parameters)
- **Training**: Custom training from scratch
- **Epochs**: 2 (optimized for 5-hour timeframe)
- **Batch Size**: 24 (GPU optimized)
- **Learning Rate**: 3e-5
- **Training Time**: 1.5 hours
- **GPU**: NVIDIA RTX 4060 with CUDA

### Random Forest Parameters
- **Algorithm**: RandomForestClassifier
- **Trees**: 200 estimators
- **Max Depth**: 400 (aggressive tuning)
- **Class Weight**: Balanced for handling imbalance

## Results

### BERT Model (State-of-the-Art)
- **Overall Accuracy**: 97.50%
- **Precision**: 98% (negative), 96% (neutral), 98% (positive)
- **Recall**: 98% (negative), 95% (neutral), 98% (positive)
- **F1-Score**: 98% (negative), 96% (neutral), 98% (positive)

### SVM Model (Traditional ML Excellence)
- **Overall Accuracy**: 91.22%
- **Precision**: High across positive, negative, and neutral classes
- **Recall**: Balanced performance for all sentiment types
- **F1-Score**: Strong harmonic mean of precision and recall

### Random Forest Model
- **Overall Accuracy**: 82.00%
- **Class Performance**: 
  - Neutral: ~5% (severe imbalance issues)
  - Positive: ~55% (moderate performance)
  - Negative: ~90% (good performance)

## Learning Outcomes

This project demonstrates:
- Complete ML pipeline from data preprocessing to model evaluation
- Feature engineering techniques for text classification
- Model selection and hyperparameter tuning
- Performance optimization strategies
- Interactive development with Jupyter notebooks
- Model comparison and understanding why certain algorithms work better for specific data types
- Deep learning implementation with custom BERT training
- GPU acceleration and optimization techniques
- Progression from traditional ML to state-of-the-art deep learning
- Cloud deployment pipeline design for production systems
- **Production deployment** with AWS EC2 and Flask API
- **Security implementation** with rate limiting and monitoring
- **Cost management** with free tier optimization and budget alerts

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the [MIT License](LICENSE).

## Cloud Deployment - AWS EC2

### Deployment Strategy
This project is designed for **AWS EC2 deployment** with the following architecture:

#### **Recommended EC2 Configuration:**
- **Instance Type**: t3.medium or t3.large
- **CPU**: 2-4 vCPUs
- **Memory**: 4-8GB RAM
- **Storage**: 20GB+ SSD
- **OS**: Ubuntu 20.04 LTS

#### **Deployment Pipeline:**
1. **Model Training**: Local development → Train BERT model (97.50% accuracy)
2. **Model Export**: Save trained model checkpoints
3. **EC2 Setup**: Launch instance, install dependencies
4. **Model Deployment**: Upload model files to EC2
5. **API Development**: Flask/FastAPI for sentiment analysis endpoint
6. **Load Balancing**: Handle multiple concurrent requests
7. **Monitoring**: Performance tracking and model health

#### **Actual Performance on EC2:**
- **Inference Time**: <2 seconds per comment (CPU-based)
- **Concurrent Requests**: 10 requests/minute (rate limited)
- **Memory Usage**: ~4GB RAM (BERT model loaded)
- **Startup Time**: 30-60 seconds (model loading)
- **API Endpoint**: `http://16.16.187.62:5000/analyze`
- **Health Check**: `http://16.16.187.62:5000/health`

#### **Production Features:**
- **RESTful API**: `/analyze` endpoint for sentiment analysis
- **Rate Limiting**: 10 requests/minute per IP address
- **Error Handling**: Graceful failure management
- **Logging**: Request/response logging with `api.log`
- **Security**: Rate limiting protection against abuse
- **Background Process**: 24/7 operation with `nohup`
- **CORS Enabled**: Ready for frontend integration

### Current Status
- ✅ **Model Development**: Complete (97.50% BERT accuracy)
- ✅ **Performance Optimization**: GPU training completed
- ✅ **EC2 Deployment**: Complete (m7i-flex.large instance)
- ✅ **API Development**: Complete (Flask server with rate limiting)
- ✅ **Production Testing**: Complete (API live and tested)
- ✅ **Security Implementation**: Rate limiting (10 requests/minute)
- ✅ **Cost Management**: Budget alerts and free tier monitoring

## Future Enhancements

- **Vercel Frontend**: React/Next.js web application
- **API Key Authentication**: Secure access control
- **Advanced Rate Limiting**: User-based quotas
- **Multi-language Support**: Extend to other languages
- **Advanced Analytics**: Sentiment trends and insights
- **Mobile App**: iOS/Android sentiment analysis app
- **AWS Lambda Integration**: Serverless sentiment analysis
- **Real-time Processing**: Stream processing capabilities

---

*Built as a learning project to understand machine learning pipelines, text classification techniques, deep learning implementation, and cloud deployment strategies.* 