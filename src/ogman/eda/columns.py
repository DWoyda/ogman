import re
import json
import unicodedata
from pathlib import Path
from typing import Optional, Iterable, Dict
import pandas as pd


def clean_columns(df: pd.DataFrame,
                  overrides: Optional[Dict[str, str]] = None,
                  save_map_to: Optional[str] = None,
                  digit_prefix: str = "col_",
                  conflict_suffix: str = "_col",
                  flatten_multiindex: bool = True,
                  mi_joiner: str = "__",
                  max_len: Optional[int] = None,
                  extra_reserved: Optional[Iterable[str]] = None,
                  return_mapping: bool = False):
    """
    Cleans and standardizes DataFrame column names.

    - Removes accents, special characters, and spaces
    - Converts CamelCase / PascalCase to snake_case
    - Forces lowercase and ensures uniqueness
    - Flattens MultiIndex columns (optional)
    - Optionally saves the column mapping (original → cleaned) to a JSON file

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    overrides : dict, optional
        Manual rename overrides applied after cleaning.
    save_map_to : str, optional
        Path to save the column name mapping as a JSON file.
    digit_prefix : str, default "col_"
        Prefix used if a column name starts with a digit.
    conflict_suffix : str, default "_col"
        Suffix added if a cleaned name conflicts with reserved names.
    flatten_multiindex : bool, default True
        Whether to flatten MultiIndex columns.
    mi_joiner : str, default "__"
        Separator used when flattening MultiIndex columns.
    max_len : int, optional
        Maximum allowed name length (names will be truncated if longer).
    extra_reserved : Iterable[str], optional
        Additional reserved names that cannot be used.
    return_mapping : bool, default False
        If True, returns both the cleaned DataFrame and the name mapping.

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, dict)
        Cleaned DataFrame, optionally with the column name mapping.
    """

    def deaccent(s: str) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFKD', str(s))
            if not unicodedata.combining(c)
        )

    def to_snake(s: str) -> str:
        s = deaccent(s).strip()
        # Camel/PascalCase -> snake_case
        s = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', s)
        s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
        s = s.replace('-', '_').replace('/', '_')
        s = re.sub(r'\s+', '_', s)
        s = s.lower()
        s = re.sub(r'[^a-z0-9_]', '', s)
        s = re.sub(r'_+', '_', s).strip('_')
        if not s:
            s = "col"
        if s[0].isdigit():
            s = f"{digit_prefix}{s}"
        return s

    def shorten(name: str, keep_for_suffix: int = 0) -> str:
        if max_len is None or len(name) <= max_len:
            return name
        cut = max_len - keep_for_suffix
        return name[:max(1, cut)]

    # 0. MultiIndex -> flat
    cols_in = (
        [mi_joiner.join(map(str, t)) for t in df.columns]
        if flatten_multiindex and isinstance(df.columns, pd.MultiIndex)
        else list(map(str, df.columns))
    )

    # 1. reserved (minimum set)
    reserved = {"index", "columns"}
    if extra_reserved:
        reserved |= set(map(str, extra_reserved))

    mapping, used = {}, set()

    # 2. cleaning + anti-collisions + uniqueness
    for orig, raw in zip(map(str, df.columns), cols_in):
        s = to_snake(raw)
        base = s
        while s in reserved:
            s = f"{base}{conflict_suffix}"
            base = s
        s = shorten(s)
        base = s
        i = 2
        while s in used:
            suf = f"_{i}"
            s = shorten(base, keep_for_suffix=len(suf)) + suf
            i += 1
        used.add(s)
        mapping[orig] = s

    df2 = df.rename(columns=mapping)

    # 3. overrides (after cleaning)
    if overrides:
        cleaned_over = {}
        inv = {v: k for k, v in mapping.items()}
        for orig_name, wanted in overrides.items():
            current = mapping.get(orig_name, to_snake(orig_name))
            vv = to_snake(wanted)
            base = vv
            while vv in reserved:
                vv = f"{base}{conflict_suffix}"
                base = vv
            base = vv
            i = 2
            while vv in (used - {current}):
                suf = f"_{i}"
                vv = shorten(base, keep_for_suffix=len(suf)) + suf
                i += 1
            used.add(vv)
            cleaned_over[current] = vv
            if orig_name in mapping:
                mapping[orig_name] = vv
        df2 = df2.rename(columns=cleaned_over)

    if save_map_to:
        Path(save_map_to).parent.mkdir(parents=True, exist_ok=True)
        with open(save_map_to, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

    return (df2, mapping) if return_mapping else df2
