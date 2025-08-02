import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader

cleaned_data = pd.read_csv('../cleaned_data.csv')
cleaned_data = cleaned_data.dropna()

# Check data distribution
sentiment_count = cleaned_data['sentiment'].value_counts()
total = len(cleaned_data)
lowest_emotion_num = sentiment_count.min()
imbalance = lowest_emotion_num / total

print(f"Data loaded: {total} comments")
print(f"Sentiment distribution: {sentiment_count.to_dict()}")
print(f"Class imbalance ratio: {imbalance:.3f}")


print("\nSetting up custom BERT model...")

# Choose BERT model
model_name = "bert-base-uncased"  # Standard BERT

try:
    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully!")
    
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3  # 3 classes: negative, neutral, positive
    )
    print("Model loaded successfully!")
    
    print(f"{model_name} loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you have internet connection for first-time download.")
    raise e  # Re-raise the exception to see the full error



print("\nPreparing data for BERT training...")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(cleaned_data['sentiment'])

print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_data['comment_text'], y_encoded,
    test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set: {len(X_train)} comments")
print(f"Testing set: {len(X_test)} comments")



print("\nCreating custom dataset for BERT...")

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
train_dataset = SentimentDataset(X_train, y_train, tokenizer)
test_dataset = SentimentDataset(X_test, y_test, tokenizer)

print(f" Training dataset: {len(train_dataset)} samples")
print(f" Testing dataset: {len(test_dataset)} samples")



print("\nTraining BERT model...")

# Training arguments
training_args = TrainingArguments(
    output_dir="./bert_sentiment_model",
    num_train_epochs=2,              # 2 epochs for 5-hour training
    per_device_train_batch_size=24,  # Increased batch size for speed
    per_device_eval_batch_size=24,   # Increased batch size
    warmup_steps=150,                # Optimized warmup
    weight_decay=0.01,               # Weight decay for regularization
    logging_dir="./logs",
    logging_steps=100,               # Less frequent logging for speed
    eval_steps=750,                  # Balanced evaluation frequency
    save_steps=750,                  # Balanced save frequency
    learning_rate=3e-5,              # Slightly higher for faster convergence
    save_total_limit=2,              # Save best 2 models
    dataloader_pin_memory=True,      # Enable pin_memory for GPU
    fp16=True,                       # Enable mixed precision (faster training)
    gradient_accumulation_steps=1,   # No gradient accumulation for speed
    dataloader_num_workers=0,        # Single worker (no multiprocessing issues)
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
print("Starting BERT training...")
trainer.train()

print("BERT training completed!")


print("\nEvaluating BERT model...")

# Evaluate on test set
eval_results = trainer.evaluate()
print(f"BERT Evaluation Results: {eval_results}")

# Check what metrics are available
print(f"Available metrics: {list(eval_results.keys())}")

# Make predictions on test set
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = y_test

# Calculate accuracy
bert_accuracy = accuracy_score(y_true, y_pred)
print(f"BERT Final Accuracy: {bert_accuracy:.4f} ({bert_accuracy*100:.2f}%)")

# Classification report
print(f"\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print(f"\nConfusion Matrix:")
print(conf_matrix)


print("\n📊 Comparing BERT with traditional methods...")

# Your actual traditional model results
traditional_results = {
    "SVM + TF-IDF": 0.9122,
    "Random Forest": 0.82,
    "XGBoost": 0.8062,
    "Naive Bayes": 0.7078,
    "Custom BERT": bert_accuracy  # Your actual BERT result
}

print("Performance Comparison:")
print("=" * 50)
for method, accuracy in traditional_results.items():
    print(f"{method:20} | {accuracy:.4f} ({accuracy*100:.2f}%)")