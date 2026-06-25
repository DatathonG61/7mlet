"""Inicializador do pacote bankmarketing."""

__version__ = "0.1.0"
__author__ = "DatathonG61"

from bankmarketing.app import app
from bankmarketing.contracts import ClientContext, DecisionResponse
from bankmarketing.data import load_modeling_table, save_modeling_table
from bankmarketing.decision_log import log_decision, retrieve_log
from bankmarketing.policies import BaselinePolicy, get_policy
from bankmarketing.synthetic import (
    OFFER_CATALOG,
    OFFER_IDS,
    assign_segments,
    generate_all,
)

__all__ = [
    "app",
    "ClientContext",
    "DecisionResponse",
    "load_modeling_table",
    "save_modeling_table",
    "log_decision",
    "retrieve_log",
    "BaselinePolicy",
    "get_policy",
    "OFFER_CATALOG",
    "OFFER_IDS",
    "assign_segments",
    "generate_all",
]
