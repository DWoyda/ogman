"""
ogman.eda
---------
EDA (Exploratory Data Analysis) utilities for data cleaning and reporting.
"""

from .columns import clean_columns
from .summary import summarize_df

__all__ = ["clean_columns", "summarize_df"]
