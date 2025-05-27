import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle


df = pd.read_csv("spam_or_not_spam.csv")

# Handle missing text
df['email'] = df['email'].fillna("")

# Check class balance
print("Label counts:\n", df['label'].value_counts())


X_train, X_test, y_train, y_test = train_test_split(
    df['email'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
)


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=3,          
        max_df=0.95         
    )),
    ('clf', LogisticRegression(class_weight='balanced'))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained pipeline
with open("spam_classifier.pkl", "wb") as f:
    pickle.dump(pipeline, f)
