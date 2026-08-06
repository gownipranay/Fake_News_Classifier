"""CLI entry point: analyze a bank statement PDF and print a JSON report.

Usage:
    python analyze.py path/to/statement.pdf [--model gemini-2.0-flash]
"""

import argparse
import json
import sys

from dotenv import load_dotenv

from src.analyzer import analyze
from src.gemini_client import DEFAULT_MODEL, extract_statement
from src.pdf_loader import load_statement_text


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Analyze a bank statement with Gemini")
    parser.add_argument("statement", help="Path to a statement PDF or text file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model id")
    args = parser.parse_args()

    text = load_statement_text(args.statement)
    statement = extract_statement(text, model=args.model)
    report = analyze(statement)

    print(json.dumps(report.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
