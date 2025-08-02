# =============================================================================
# SIMPLE NAIVE BAYES SENTIMENT ANALYSIS
# =============================================================================
# This file implements a basic Naive Bayes classifier with TF-IDF features
# for sentiment analysis. Organized into blocks for Jupyter notebook use.
# =============================================================================

# =============================================================================
# BLOCK 1: IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# BLOCK 2: DATA LOADING AND EXPLORATION
# =============================================================================

cleaned_data = pd.read_csv('../cleaned_data.csv')
cleaned_data = cleaned_data.dropna()

sentiment_count = cleaned_data['sentiment'].value_counts()
total = len(cleaned_data)
lowest_emotion_num = sentiment_count.min()
imbalance = lowest_emotion_num / total

print(f"Data loaded: {total} comments")
print(f"Sentiment distribution: {sentiment_count.to_dict()}")
print(f"Class imbalance ratio: {imbalance:.3f}")


# =============================================================================
# BLOCK 3: FEATURE ENGINEERING (HEAVY COMPUTATION - RUN ONCE)
# =============================================================================

# Simple TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,        # Keep top 10,000 words
    stop_words='english',      # Remove common words
    ngram_range=(1, 2),        # Use 1-word and 2-word combinations
    min_df=2,                  # Word must appear in at least 2 documents
    max_df=0.95,               # Word must not appear in more than 95% of documents
    sublinear_tf=True          # Apply sublinear scaling
)

# Transform text to numerical features
X = tfidf_vectorizer.fit_transform(cleaned_data['comment_text'])
y = cleaned_data['sentiment']

print(f"Feature matrix shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")

# Show some feature names
feature_names = tfidf_vectorizer.get_feature_names_out()
print(f"Sample features: {feature_names[:10]}")

# =============================================================================
# BLOCK 4: TRAIN-TEST SPLIT
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,           # 20% for testing
    random_state=42,         # For reproducible results
    stratify=y               # Maintain class proportions
)

print(f"Training set: {X_train.shape[0]} comments")
print(f"Testing set: {X_test.shape[0]} comments")

# Check class distribution in splits
print(f"\nTraining set class distribution:")
print(y_train.value_counts(normalize=True) * 100)
print(f"\nTesting set class distribution:")
print(y_test.value_counts(normalize=True) * 100)

# =============================================================================
# BLOCK 5: MODEL TRAINING (PARAMETER TUNING ZONE)
# =============================================================================

# Try different Naive Bayes variants
models = {
    'MultinomialNB': MultinomialNB(alpha=1.0),
    'ComplementNB': ComplementNB(alpha=1.0),    # Often better for imbalanced data
    'BernoulliNB': BernoulliNB(alpha=1.0)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"{name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Find best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n🏆 Best model: {best_model_name} with {results[best_model_name]:.4f} accuracy")

# =============================================================================
# BLOCK 6: MODEL EVALUATION
# =============================================================================
print("\n📈 Detailed evaluation of best model...")

# Use best model for detailed evaluation
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Final {best_model_name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# =============================================================================
# BLOCK 7: FEATURE IMPORTANCE ANALYSIS (NAIVE BAYES EXCLUSIVE!)
# =============================================================================
print("\n🔍 Analyzing feature importance...")

# Get feature importance for each class
feature_names = tfidf_vectorizer.get_feature_names_out()

print(f"Top 10 most important features for each sentiment:")

for i, class_name in enumerate(['negative', 'neutral', 'positive']):
    # Get log probabilities for this class
    if hasattr(best_model, 'feature_log_prob_'):
        class_probs = best_model.feature_log_prob_[i]
        # Get top 10 features (highest probability)
        top_indices = np.argsort(class_probs)[-10:][::-1]
        
        print(f"\n{class_name.upper()} sentiment:")
        for idx in top_indices:
            feature_name = feature_names[idx]
            prob = class_probs[idx]
            print(f"  '{feature_name}': {prob:.3f}")

# =============================================================================
# BLOCK 8: VISUALIZATIONS
# =============================================================================
print("\n📊 Creating visualizations...")

# Set up the plotting style
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Model Comparison
axes[0, 0].bar(results.keys(), results.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0, 0].set_title('Naive Bayes Model Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_ylim(0, 1)
for i, v in enumerate(results.values()):
    axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

# 2. Confusion Matrix Heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive'],
            ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# 3. Class Distribution
sentiment_counts = cleaned_data['sentiment'].value_counts()
axes[1, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
               colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[1, 0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')

# 4. Accuracy by Class
class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
axes[1, 1].bar(['negative', 'neutral', 'positive'], class_accuracy, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[1, 1].set_title('Accuracy by Class', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_ylim(0, 1)
for i, v in enumerate(class_accuracy):
    axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('naive_bayes_results.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# BLOCK 9: FINAL SUMMARY
# =============================================================================
print("\n" + "="*60)
print("🎯 FINAL SUMMARY")
print("="*60)

print(f"📊 Dataset: {total} comments")
print(f"🔧 Features: {X.shape[1]} TF-IDF features")
print(f"🏆 Best Model: {best_model_name}")
print(f"📈 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\n📋 Model Performance:")
for name, acc in results.items():
    print(f"  {name}: {acc:.4f} ({acc*100:.2f}%)")

print(f"\n🎯 Key Insights:")
print(f"  • Naive Bayes with TF-IDF achieved {accuracy*100:.1f}% accuracy")
print(f"  • {best_model_name} performed best among NB variants")
print(f"  • Feature importance analysis shows which words matter most")
print(f"  • Model is fast to train and interpretable")

print(f"\n💡 Next Steps:")
print(f"  • Try different alpha values (hyperparameter tuning)")
print(f"  • Experiment with more features (increase max_features)")
print(f"  • Combine with character-level features")
print(f"  • Try ensemble with SVM and Random Forest")

print("="*60) 