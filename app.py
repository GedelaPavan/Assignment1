import gradio as gr
from utils import mask_text
from models import load_model
import os

os.system("python -m spacy download en_core_web_sm")
model = load_model()

def classify_email(email_body):
    masked_email, entities = mask_text(email_body)
    category = model.predict([masked_email])[0]
    return {
        "input_email_body": email_body,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }

interface = gr.Interface(
    fn=classify_email,
    inputs=gr.Textbox(lines=10, label="Email Body"),
    outputs="json",
    title="Akaike Email Classifier with PII Masking"
)

interface.launch()
