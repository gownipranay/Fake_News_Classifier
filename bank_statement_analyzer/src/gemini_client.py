"""Thin wrapper around the google-genai SDK.

Supports both Vertex AI (set GOOGLE_GENAI_USE_VERTEXAI=true plus project
and location) and the Gemini Developer API (set GEMINI_API_KEY). The
google-genai client picks the backend up from those environment
variables automatically.
"""

import os

from google import genai
from google.genai import types

from .models import Statement

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

EXTRACTION_PROMPT = """\
You are a meticulous financial-document parser. Extract the bank
statement below into structured data.

Rules:
- Every transaction line becomes one entry. Amounts are positive
  numbers; use type "credit" for money in and "debit" for money out.
- Dates must be ISO format (YYYY-MM-DD). Infer the year from the
  statement period when a line omits it.
- Copy the account number only in the masked form printed on the
  statement; never reconstruct full numbers.
- Guess a category per transaction (salary, transfer, atm, fees,
  utilities, shopping, rent, emi, other).
- If a field is not present in the statement, leave it null. Do not
  invent values.

Statement text:
---
{statement_text}
---
"""


def extract_statement(statement_text: str, model: str = DEFAULT_MODEL) -> Statement:
    """Ask Gemini to parse raw statement text into a Statement object."""
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=EXTRACTION_PROMPT.format(statement_text=statement_text),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Statement,
            temperature=0,
        ),
    )
    parsed = response.parsed
    if isinstance(parsed, Statement):
        return parsed
    # Fall back to validating the raw JSON text ourselves.
    return Statement.model_validate_json(response.text)
