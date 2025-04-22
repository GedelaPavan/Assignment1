from fastapi import FastAPI, Request
from pydantic import BaseModel
from utils import mask_text
from models import load_model

app = FastAPI()
model = load_model()

class EmailInput(BaseModel):
    email_body: str

@app.post("/")
def classify_email(input: EmailInput):
    original_email = input.email_body
    masked_email, entities = mask_text(original_email)
    category = model.predict([masked_email])[0]

    return {
        "input_email_body": original_email,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }
