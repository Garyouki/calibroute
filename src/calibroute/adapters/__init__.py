"""Adapters from task-specific prediction logs to CalibRoute records."""

from .financial_ner import read_financial_ner_records, write_records_csv

__all__ = ["read_financial_ner_records", "write_records_csv"]
