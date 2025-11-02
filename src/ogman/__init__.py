"""
Modules:
- ogman.web_scraping: download data (in progress)
- ogman.eda: EDA tools (cleaning, summarization)
- ogman.visualization: visualization tools (in progress)
- ogman.ml: machine learning (in progress)
"""

from .eda import clean_columns, summarize_df

__all__ = ["clean_columns", "summarize_df"]
