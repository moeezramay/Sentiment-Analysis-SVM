# BERT Sentiment Analysis

This folder contains the custom BERT implementation for sentiment analysis achieving **97.50% accuracy**.

## Important Note: Large Model Files

The trained BERT model checkpoints are **not included** in this repository due to size limitations (>100MB). 

### How to Use:

1. **Train the model yourself:**
   ```bash
   python simple_bert.py
   ```
   This will train the BERT model from scratch (takes ~1.5 hours).

2. **Or use the evaluation script:**
   ```bash
   python evaluate_bert.py
   ```
   This will automatically download the pre-trained BERT model and evaluate it.

### Model Files:
- `simple_bert.py` - Complete BERT training implementation
- `evaluate_bert.py` - Evaluation script for trained model
- `bert_sentiment_model/` - Trained model checkpoints (generated during training)

### Expected Results:
- **Training Time:** ~1.5 hours
- **Final Accuracy:** 97.50%
- **Model Size:** ~440MB (checkpoints)

### Alternative: Use Pre-trained Model
If you don't want to train from scratch, the evaluation script will automatically download the pre-trained BERT model when you run it. 