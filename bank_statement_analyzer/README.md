# Bank Statement Analyzer

Analyzes bank statement PDFs with Google's Gemini models: extracts every
transaction into structured data, estimates monthly income, finds
recurring deposits, and flags risk signals (overdrafts, fee/returned
payments, unusual or round-figure deposits, balance mismatches that can
indicate missing pages or tampering).

## How it works

1. **`src/pdf_loader.py`** — pulls raw text from the PDF (pypdf).
   Scanned statements need OCR first.
2. **`src/gemini_client.py`** — sends the text to Gemini with a JSON
   response schema (`src/models.py`), getting back typed transactions
   and statement metadata. Runs on Vertex AI or the Gemini Developer
   API — the `google-genai` SDK picks the backend from environment
   variables.
3. **`src/analyzer.py`** — deterministic Python over the extracted
   data: totals, income estimate normalized by statement period,
   recurring-deposit detection, and rule-based risk signals. No LLM in
   this step, so the numbers are reproducible.

## Setup

```bash
cd bank_statement_analyzer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your project / API key
```

For Vertex AI, authenticate with `gcloud auth application-default login`
(or a service account) and set `GOOGLE_CLOUD_PROJECT` and
`GOOGLE_CLOUD_LOCATION` in `.env`. For the Developer API, set
`GEMINI_API_KEY` instead.

## Usage

CLI — prints the full report as JSON:

```bash
python analyze.py path/to/statement.pdf
```

Web UI:

```bash
streamlit run app.py
```

## Notes

- Statements contain sensitive data. `.gitignore` excludes `.env`,
  `statements/`, and `*.pdf` so documents and credentials never get
  committed. Statement text is sent to the Gemini API you configured —
  use Vertex AI in your own GCP project if data residency matters.
- Risk signals are heuristics to prioritize human review, not a credit
  decision.
