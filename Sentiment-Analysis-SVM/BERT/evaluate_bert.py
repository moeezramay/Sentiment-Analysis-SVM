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
from torch.utils.data import Dataset

print("Loading data...")
cleaned_data = pd.read_csv('../cleaned_data.csv')
cleaned_data = cleaned_data.dropna()

print("Loading trained BERT model...")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the trained model
model = AutoModelForSequenceClassification.from_pretrained(
    "./bert_sentiment_model/checkpoint-10138",  # Load the final checkpoint
    num_labels=3
)

print("Preparing data...")
# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(cleaned_data['sentiment'])

# Split data (same as training)
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_data['comment_text'], y_encoded,
    test_size=0.2, random_state=42, stratify=y_encoded
)

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
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create test dataset
test_dataset = SentimentDataset(X_test, y_test, tokenizer)

print("Evaluating BERT model...")
# Setup trainer for evaluation
training_args = TrainingArguments(
    output_dir="./temp_eval",
    per_device_eval_batch_size=32,
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
)

# Make predictions
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = y_test

# Calculate accuracy
bert_accuracy = accuracy_score(y_true, y_pred)
print(f"\n🎯 BERT Final Accuracy: {bert_accuracy:.4f} ({bert_accuracy*100:.2f}%)")

# Classification report
print(f"\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print(f"\n📈 Confusion Matrix:")
print(conf_matrix)

# Compare with traditional methods
traditional_results = {
    "SVM + TF-IDF": 0.9122,
    "Random Forest": 0.82,
    "XGBoost": 0.8062,
    "Naive Bayes": 0.7078,
    "Custom BERT": bert_accuracy
}

print(f"\n🏆 Performance Comparison:")
print("=" * 50)
for method, accuracy in traditional_results.items():
    print(f"{method:20} | {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\n🎉 BERT Improvement over SVM: {((bert_accuracy - 0.9122) * 100):.2f}%") 