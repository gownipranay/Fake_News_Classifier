"""Turn an extracted Statement into an AnalysisReport.

All the analysis here is deterministic Python over the structured data
Gemini produced — income estimation, deposit patterns, and rule-based
risk signals a lender or underwriter would care about.
"""

from collections import defaultdict

from .models import (
    AnalysisReport,
    RiskSignal,
    Statement,
    Transaction,
    TransactionType,
)

INCOME_CATEGORIES = {"salary"}
LARGE_DEPOSIT_MULTIPLIER = 3.0
ROUND_AMOUNT_THRESHOLD = 10_000


def _months_covered(statement: Statement) -> float:
    meta = statement.metadata
    if meta.period_start and meta.period_end:
        days = (meta.period_end - meta.period_start).days
        return max(days / 30.44, 1 / 30.44)
    dates = [t.date for t in statement.transactions]
    if not dates:
        return 1.0
    return max((max(dates) - min(dates)).days / 30.44, 1 / 30.44)


def _recurring_deposits(credits: list[Transaction]) -> list[str]:
    """Descriptions that credit the account in 2+ distinct months."""
    months_by_desc: dict[str, set[str]] = defaultdict(set)
    for t in credits:
        key = t.description.strip().lower()
        months_by_desc[key].add(t.date.strftime("%Y-%m"))
    return sorted(desc for desc, months in months_by_desc.items() if len(months) >= 2)


def _risk_signals(statement: Statement, credits: list[Transaction]) -> list[RiskSignal]:
    signals: list[RiskSignal] = []

    balances = [t.balance for t in statement.transactions if t.balance is not None]
    negatives = [b for b in balances if b < 0]
    if negatives:
        signals.append(
            RiskSignal(
                code="overdraft",
                severity="high",
                detail=f"Balance went negative {len(negatives)} time(s); "
                f"lowest {min(negatives):,.2f}.",
            )
        )

    fee_hits = [
        t
        for t in statement.transactions
        if t.type is TransactionType.DEBIT
        and (t.category == "fees" or "return" in t.description.lower())
    ]
    if fee_hits:
        signals.append(
            RiskSignal(
                code="fees_or_returns",
                severity="medium",
                detail=f"{len(fee_hits)} fee/returned-payment debit(s) on the statement.",
            )
        )

    if credits:
        avg_credit = sum(t.amount for t in credits) / len(credits)
        spikes = [t for t in credits if t.amount > LARGE_DEPOSIT_MULTIPLIER * avg_credit]
        for t in spikes:
            signals.append(
                RiskSignal(
                    code="unusual_deposit",
                    severity="medium",
                    detail=f"{t.date} deposit of {t.amount:,.2f} "
                    f"({t.description}) is >{LARGE_DEPOSIT_MULTIPLIER:.0f}x "
                    "the average credit.",
                )
            )

        round_large = [
            t
            for t in credits
            if t.amount >= ROUND_AMOUNT_THRESHOLD and t.amount == round(t.amount, -3)
        ]
        if round_large:
            signals.append(
                RiskSignal(
                    code="round_amount_deposits",
                    severity="low",
                    detail=f"{len(round_large)} large round-figure deposit(s); "
                    "worth verifying the source.",
                )
            )

    meta = statement.metadata
    if (
        meta.opening_balance is not None
        and meta.closing_balance is not None
        and balances
    ):
        expected = meta.opening_balance + sum(
            t.amount if t.type is TransactionType.CREDIT else -t.amount
            for t in statement.transactions
        )
        if abs(expected - meta.closing_balance) > 1.0:
            signals.append(
                RiskSignal(
                    code="balance_mismatch",
                    severity="high",
                    detail=f"Opening balance plus transactions gives {expected:,.2f} "
                    f"but the statement closes at {meta.closing_balance:,.2f}. "
                    "Possible missing pages or a tampered document.",
                )
            )

    return signals


def analyze(statement: Statement) -> AnalysisReport:
    credits = [t for t in statement.transactions if t.type is TransactionType.CREDIT]
    debits = [t for t in statement.transactions if t.type is TransactionType.DEBIT]

    total_credits = sum(t.amount for t in credits)
    total_debits = sum(t.amount for t in debits)

    months = _months_covered(statement)
    income_credits = [t for t in credits if (t.category or "").lower() in INCOME_CATEGORIES]
    income_base = income_credits if income_credits else credits
    estimated_monthly_income = sum(t.amount for t in income_base) / months

    return AnalysisReport(
        metadata=statement.metadata,
        total_credits=round(total_credits, 2),
        total_debits=round(total_debits, 2),
        net_flow=round(total_credits - total_debits, 2),
        estimated_monthly_income=round(estimated_monthly_income, 2),
        recurring_deposits=_recurring_deposits(credits),
        largest_deposits=sorted(credits, key=lambda t: t.amount, reverse=True)[:5],
        risk_signals=_risk_signals(statement, credits),
    )
