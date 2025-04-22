import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def train_model(X, y):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", RandomForestClassifier(n_estimators=50,max_depth=10))
    ])
    pipeline.fit(X, y)

    with open("model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)
