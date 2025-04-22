import pandas as pd
from models import train_model

# Load your dataset
df = pd.read_csv("data/emails.csv")

# Optional: You can apply masking here if needed before training
X = df["email"]
y = df["type"]

# Train and save the model
train_model(X, y)
print("Model trained and saved to model.pkl!")
