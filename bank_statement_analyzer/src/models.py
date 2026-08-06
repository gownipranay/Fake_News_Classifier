"""Pydantic schemas shared by the extractor and analyzer."""

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class TransactionType(str, Enum):
    CREDIT = "credit"
    DEBIT = "debit"


class Transaction(BaseModel):
    """A single line item parsed from a bank statement."""

    date: date
    description: str
    amount: float = Field(gt=0, description="Absolute amount; direction comes from type")
    type: TransactionType
    balance: float | None = Field(default=None, description="Running balance if shown")
    category: str | None = Field(
        default=None,
        description="Best-guess category, e.g. salary, transfer, atm, fees, utilities",
    )


class StatementMetadata(BaseModel):
    account_holder: str | None = None
    account_number_masked: str | None = Field(
        default=None, description="Only the masked form printed on the statement"
    )
    bank_name: str | None = None
    period_start: date | None = None
    period_end: date | None = None
    opening_balance: float | None = None
    closing_balance: float | None = None
    currency: str | None = None


class Statement(BaseModel):
    """Full structured extraction of one statement document."""

    metadata: StatementMetadata
    transactions: list[Transaction]


class RiskSignal(BaseModel):
    code: str
    severity: str = Field(description="low, medium or high")
    detail: str


class AnalysisReport(BaseModel):
    metadata: StatementMetadata
    total_credits: float
    total_debits: float
    net_flow: float
    estimated_monthly_income: float
    recurring_deposits: list[str]
    largest_deposits: list[Transaction]
    risk_signals: list[RiskSignal]
