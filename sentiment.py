import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt

cleaned_data = pd.read_csv('cleaned_data.csv')

cleaned_data = cleaned_data.dropna()  #remove null values

sentiment_count = cleaned_data['sentiment'].value_counts()

#checking for imabalance----------------------------------

total = len(cleaned_data)
lowest_emotion_num = sentiment_count.min()
imbalance = lowest_emotion_num / total

#it gave greater than 0.2 so we good-----------------------



#Convert text to vectors-----------------------------------
print("Step 1: Converting text to numerical vectors:")

# Combine word-level and character-level features
from sklearn.pipeline import FeatureUnion

# Word-level features
word_vectorizer = TfidfVectorizer(
    max_features=12000,  # Word features
    stop_words='english',
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

# Character-level features (captures patterns like "!!!" or "???")
char_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 5),  # 3-5 character combinations
    max_features=3000,
    min_df=2,
    max_df=0.95
)

#s
# Combine both feature types
feature_union = FeatureUnion([
    ('word_features', word_vectorizer),
    ('char_features', char_vectorizer)
])

print("Creating combined word and character features...")

# vectors stored in X
X = feature_union.fit_transform(cleaned_data['comment_text'])

# sentiment labels stored in y
y = cleaned_data['sentiment']

print("\nStep 2: Splitting data into training and testing sets...")

#spliting 80-20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # 20% for testing
    random_state=42,  # For reproducible results
    stratify=y  # Maintain same proportion of sentiments in both sets
)

print(f"Training set: {X_train.shape[0]} comments")
print(f"Testing set: {X_test.shape[0]} comments")
print(f"Training set sentiment distribution:")
print(y_train.value_counts(normalize=True) * 100)

# Apply SMOTE to balance the training data
from imblearn.over_sampling import SMOTE
print("\nApplying SMOTE to balance classes...")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - Training set: {X_train_balanced.shape[0]} comments")
print("Balanced training set sentiment distribution:")
print(pd.Series(y_train_balanced).value_counts(normalize=True) * 100)

#print(y_train.value_counts(normalize=True) * 100) #check if it matches the original data, y_train is the training set for emotion labels

# Step 3: Model Training
print("\nStep 3: Training the sentiment analysis model...")

# Create and train the model using SVM
from sklearn.svm import LinearSVC
model = LinearSVC(
    random_state=42,
    max_iter=5000,    # Increased from 2000 to 5000 for better convergence
    C=2.0,           # Regularization parameter
    loss='hinge'     # Standard loss function for SVM
)
model.fit(X_train_balanced, y_train_balanced)  # Use balanced data


#----------------------------------------------------------Checking the model----------------------------------------------------------

print("Model training completed!")
print(f"Number of features: {model.n_features_in_}")

# Step 4: Model Evaluation
print("\nStep 4: Evaluating model performance...")

# Make predictions on test data
y_predict = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predict)
print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Detailed performance report
print("\nDetailed Performance Report:")
print(classification_report(y_test, y_predict))

# Confusion Matrix
print("\nConfusion Matrix:")
print("Rows: Actual, Columns: Predicted")
print(confusion_matrix(y_test, y_predict))

print(f"\nFinal SVM Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")