import pandas as pd

def summarize_df(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Generate an extended EDA summary of a DataFrame (preserves column order).

    Includes:
    - dtype
    - logical type
    - unique values
    - missing data
    - duplicates (per column)
    - min/max/mean for numeric features
    - top N frequent values for categorical
    - date ranges for datetime features

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to summarize.
    top_n : int, default=3
        Number of most frequent values to show for categorical features.

    Returns
    -------
    pd.DataFrame
        Data summary table with key statistics per column.
    """

    n_rows = len(df)
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nunique": df.nunique(dropna=False),
        "missing": df.isna().sum(),
        "missing [%]": (df.isna().mean() * 100).round(2),
    }, index=df.columns)

    # Duplikaty per kolumna
    dup_counts = df.columns.to_series().apply(lambda c: df[c].duplicated(keep=False).sum())
    dup_perc = (dup_counts / n_rows * 100).round(2)
    summary["duplicates"] = dup_counts
    summary["duplicates [%]"] = dup_perc

    # Typ logiczny kolumny
    def logical_type(s: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(s):
            return "numeric"
        elif pd.api.types.is_bool_dtype(s):
            return "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            return "datetime"
        elif pd.api.types.is_object_dtype(s):
            nunique = s.nunique(dropna=True)
            return "categorical" if nunique <= 10 else "text"
        return "other"

    summary["logical_type"] = [logical_type(df[col]) for col in df.columns]

    # Statystyki dla liczb
    num_cols = df.select_dtypes(include="number")
    summary.loc[num_cols.columns, "min"] = num_cols.min()
    summary.loc[num_cols.columns, "max"] = num_cols.max()
    summary.loc[num_cols.columns, "mean"] = num_cols.mean().round(2)

    # Top wartości dla kategorii
    cat_cols = summary.query("logical_type == 'categorical'").index
    for col in cat_cols:
        top_values = df[col].value_counts(dropna=False).head(top_n)
        summary.loc[col, "top_values"] = ", ".join([f"{i}: {v}" for i, v in zip(top_values.index, top_values.values)])

    # Zakres dat
    dt_cols = summary.query("logical_type == 'datetime'").index
    for col in dt_cols:
        summary.loc[col, "min_date"] = df[col].min()
        summary.loc[col, "max_date"] = df[col].max()

    # Kolejność kolumn
    column_order = [
        "dtype", 
        "logical_type",
        "nunique", 
        "missing", 
        "missing [%]",
        "duplicates", 
        "duplicates [%]",
        "min", 
        "max", 
        "mean",
        "top_values", 
        "min_date", 
        "max_date"
    ]
    summary = summary[[col for col in column_order if col in summary.columns]]

    return summary
