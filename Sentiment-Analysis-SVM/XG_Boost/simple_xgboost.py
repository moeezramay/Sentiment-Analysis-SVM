# =============================================================================
# SIMPLE XGBOOST SENTIMENT ANALYSIS
# =============================================================================
# This file implements XGBoost classifier with TF-IDF features
# for sentiment analysis. Organized into blocks for Jupyter notebook use.
# =============================================================================

# =============================================================================
# BLOCK 1: IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

print("✅ All imports successful!")

# =============================================================================
# BLOCK 2: DATA LOADING AND EXPLORATION
# =============================================================================
print("\n📊 Loading and exploring data...")

# Load the cleaned data
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

# Display sample comments
print("\nSample comments:")
for i, (comment, sentiment) in enumerate(zip(cleaned_data['comment_text'].head(3), cleaned_data['sentiment'].head(3))):
    print(f"{i+1}. '{comment[:50]}...' -> {sentiment}")

# =============================================================================
# BLOCK 3: FEATURE ENGINEERING (HEAVY COMPUTATION - RUN ONCE)
# =============================================================================
print("\n🔧 Creating TF-IDF features...")

# TF-IDF vectorization optimized for XGBoost
tfidf_vectorizer = TfidfVectorizer(
    max_features=15000,        # Keep top 15,000 words
    stop_words='english',      # Remove common words
    ngram_range=(1, 3),        # Use 1-word, 2-word, and 3-word combinations
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
print("\n✂️ Splitting data into training and testing sets...")

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
print("\n🤖 Training XGBoost model...")

# XGBoost model with optimized parameters
xgb_model = xgb.XGBClassifier(
    n_estimators=200,          # Number of trees
    max_depth=6,               # Maximum depth of each tree
    learning_rate=0.1,         # How much each tree contributes (eta)
    subsample=0.8,             # Use 80% of data for each tree
    colsample_bytree=0.8,      # Use 80% of features for each tree
    random_state=42,           # For reproducible results
    n_jobs=-1,                 # Use all CPU cores
    eval_metric='mlogloss'     # Multi-class log loss
)

# Train the model
print("Training XGBoost (this may take a few minutes)...")
xgb_model.fit(X_train, y_train)

print("✅ XGBoost training completed!")

# =============================================================================
# BLOCK 6: MODEL EVALUATION
# =============================================================================
print("\n📈 Evaluating XGBoost model...")

# Make predictions
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"XGBoost Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# =============================================================================
# BLOCK 7: FEATURE IMPORTANCE ANALYSIS (XGBOOST EXCLUSIVE!)
# =============================================================================
print("\n🔍 Analyzing feature importance...")

# Get feature importance scores
feature_importance = xgb_model.feature_importances_
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get top 20 most important features
top_indices = np.argsort(feature_importance)[-20:][::-1]

print("Top 20 most important features:")
for i, idx in enumerate(top_indices):
    feature_name = feature_names[idx]
    importance = feature_importance[idx]
    print(f"{i+1:2d}. '{feature_name}': {importance:.4f}")

# =============================================================================
# BLOCK 8: VISUALIZATIONS
# =============================================================================
print("\n📊 Creating visualizations...")

# Set up the plotting style
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Feature Importance Top 15
top_15_indices = np.argsort(feature_importance)[-15:][::-1]
top_15_features = [feature_names[i] for i in top_15_indices]
top_15_importance = [feature_importance[i] for i in top_15_indices]

axes[0, 0].barh(range(len(top_15_features)), top_15_importance, color='#4ECDC4')
axes[0, 0].set_yticks(range(len(top_15_features)))
axes[0, 0].set_yticklabels(top_15_features)
axes[0, 0].set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Importance Score')

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
plt.savefig('xgboost_results.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# BLOCK 9: HYPERPARAMETER TUNING (OPTIONAL)
# =============================================================================
print("\n🔧 Hyperparameter tuning options...")

# You can experiment with these parameters:
tuning_options = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.15, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

print("Available parameters to tune:")
for param, values in tuning_options.items():
    print(f"  {param}: {values}")

# Example of how to try different parameters:
print("\nExample - trying different learning rates:")
for lr in [0.05, 0.1, 0.15, 0.2]:
    model = xgb.XGBClassifier(
        n_estimators=100,  # Fewer trees for faster testing
        max_depth=6,
        learning_rate=lr,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Learning rate {lr}: {acc:.4f}")

# =============================================================================
# BLOCK 10: FINAL SUMMARY
# =============================================================================
print("\n" + "="*60)
print("🎯 FINAL SUMMARY")
print("="*60)

print(f"📊 Dataset: {total} comments")
print(f"🔧 Features: {X.shape[1]} TF-IDF features")
print(f"🏆 Model: XGBoost")
print(f"📈 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\n🎯 Key Insights:")
print(f"  • XGBoost achieved {accuracy*100:.1f}% accuracy")
print(f"  • Feature importance analysis shows which words matter most")
print(f"  • Model handles class imbalance well")
print(f"  • Fast training and prediction")

print(f"\n💡 Next Steps:")
print(f"  • Try hyperparameter tuning (learning_rate, max_depth)")
print(f"  • Experiment with more features (increase max_features)")
print(f"  • Add character-level features")
print(f"  • Try ensemble with SVM and Naive Bayes")

print("="*60) 