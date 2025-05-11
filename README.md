Email Classification Assignment

##  Objective
Classify support emails and mask PII using traditional NLP + ML techniques.

##  Deployed Demo
🔗 [View Live on Hugging Face Spaces](https://pavanstunner-emails.hf.space/?logs=container&__theme=system&deep_link=7CdwPeVqmW4)

##  Model
- TF-IDF + RandomForestClassifier
- PII masking via Regex + SpaCy NER

##  How to Run Locally
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py

📂 Project Structure
app.py - Gradio UI

utils.py - PII detection & masking

models.py - Model training & loading

model.pkl - Final trained model

train_model_script.py - Train from emails.csv
