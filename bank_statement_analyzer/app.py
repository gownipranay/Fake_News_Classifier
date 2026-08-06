"""Streamlit UI for the bank statement analyzer.

Run with:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.analyzer import analyze
from src.gemini_client import DEFAULT_MODEL, extract_statement
from src.pdf_loader import load_statement_text

load_dotenv()

st.set_page_config(page_title="Bank Statement Analyzer", page_icon="🏦")
st.title("🏦 Bank Statement Analyzer")
st.caption(
    "Upload a bank statement PDF. Gemini extracts the transactions, "
    "then deterministic rules estimate income and flag risk signals."
)

uploaded = st.file_uploader("Bank statement", type=["pdf", "txt"])
model = st.text_input("Gemini model", value=DEFAULT_MODEL)

if uploaded is not None and st.button("Analyze", type="primary"):
    suffix = ".pdf" if uploaded.name.lower().endswith(".pdf") else ".txt"
    tmp_path = f"/tmp/statement{suffix}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    with st.spinner("Extracting transactions with Gemini..."):
        text = load_statement_text(tmp_path)
        statement = extract_statement(text, model=model)
    report = analyze(statement)

    meta = report.metadata
    st.subheader("Statement")
    st.write(
        {
            "Bank": meta.bank_name,
            "Account holder": meta.account_holder,
            "Account": meta.account_number_masked,
            "Period": f"{meta.period_start} → {meta.period_end}",
            "Currency": meta.currency,
        }
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total credits", f"{report.total_credits:,.2f}")
    col2.metric("Total debits", f"{report.total_debits:,.2f}")
    col3.metric("Est. monthly income", f"{report.estimated_monthly_income:,.2f}")

    st.subheader("Risk signals")
    if report.risk_signals:
        for s in report.risk_signals:
            icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(s.severity, "⚪")
            st.write(f"{icon} **{s.code}** ({s.severity}): {s.detail}")
    else:
        st.success("No risk signals detected.")

    st.subheader("Recurring deposits")
    st.write(report.recurring_deposits or "None found across multiple months.")

    st.subheader("Transactions")
    st.dataframe(
        pd.DataFrame([t.model_dump(mode="json") for t in statement.transactions]),
        use_container_width=True,
    )
