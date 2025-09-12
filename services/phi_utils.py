# services/phi_utils.py
import re
import spacy
from typing import Tuple

# load once
_nlp = spacy.load("en_core_web_sm")

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b")

def detect_phi(text: str) -> dict:
    doc = _nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    return {"entities": entities, "emails": emails, "phones": phones}

def redact_phi(text: str) -> str:
    doc = _nlp(text)
    redacted = text
    for ent in doc.ents:
        start, end = ent.start_char, ent.end_char
        redacted = redacted[:start] + "[REDACTED]" + redacted[end:]
    # also redact emails & phones
    redacted = EMAIL_RE.sub("[REDACTED_EMAIL]", redacted)
    redacted = PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    return redacted
