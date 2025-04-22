import re
import spacy

nlp = spacy.load("en_core_web_sm")

ENTITY_PATTERNS = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone_number": r"\b(?:\+91[-\s]?)?[789]\d{9}\b",
    "aadhar_num": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}\b",
    "credit_debit_no": r"\b(?:\d[ -]*?){13,16}\b",
    "cvv_no": r"\b\d{3}\b",
    "expiry_no": r"\b(0[1-9]|1[0-2])\/?([0-9]{2}|[0-9]{4})\b",
    "dob": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
}

def mask_text(text):
    masked_entities = []
    original_text = text

    for label, pattern in ENTITY_PATTERNS.items():
        for match in re.finditer(pattern, text):
            start, end = match.span()
            entity = match.group()
            masked_entities.append({
                "position": [start, end],
                "classification": label,
                "entity": entity
            })
            text = text.replace(entity, f"[{label}]")

    # SpaCy NER for full names
    doc = nlp(original_text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            start, end = ent.start_char, ent.end_char
            entity = ent.text
            masked_entities.append({
                "position": [start, end],
                "classification": "full_name",
                "entity": entity
            })
            text = text.replace(entity, "[full_name]")

    return text, masked_entities
