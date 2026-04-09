# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Best-effort normalization of LLM-extracted temporal scope strings to ISO-8601."""

from __future__ import annotations

import re

# Quarter start months
_QUARTER_START = {"1": "01", "2": "04", "3": "07", "4": "10"}
_QUARTER_END = {"1": "03", "2": "06", "3": "09", "4": "12"}

# Month name -> number
_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "jun": "06", "jul": "07", "aug": "08", "sep": "09",
    "oct": "10", "nov": "11", "dec": "12",
    # German month names
    "januar": "01", "februar": "02", "maerz": "03", "märz": "03",
    "mai": "05", "juni": "06", "juli": "07",
    "september": "09", "oktober": "10", "dezember": "12",
}


def resolve_temporal_scope(
    scope: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Attempt to resolve a free-text temporal scope into structured dates.

    Returns
    -------
    (valid_from, valid_until, unresolved_qualifier)
        - valid_from: ISO-8601 date or None
        - valid_until: ISO-8601 date or None
        - unresolved_qualifier: The original string if it could not be resolved, else None
    """
    if not scope or not scope.strip():
        return None, None, None

    scope = scope.strip()
    lower = scope.lower()

    # "formerly" / "former" -> ended at some unspecified time
    if lower in ("formerly", "former", "previous", "previously", "past"):
        return None, None, scope

    # "current" / "present" / "ongoing" -> started at some unspecified time, still valid
    if lower in ("current", "present", "ongoing", "now"):
        return None, None, scope

    # Try year-range: "2020-2023", "2020 - 2023", "2020 to 2023"
    m = re.match(r"(\d{4})\s*[-–—to]+\s*(\d{4})", lower)
    if m:
        return f"{m.group(1)}-01-01", f"{m.group(2)}-12-31", None

    # Try year-range with "present"/"now": "2020-present", "2020 to present"
    m = re.match(r"(\d{4})\s*[-–—to]+\s*(?:present|now|current|heute)", lower)
    if m:
        return f"{m.group(1)}-01-01", None, None

    # Try "since <Month> <Year>": "since January 2024"
    m = re.match(r"(?:since|ab|seit)\s+(\w+)\s+(\d{4})", lower)
    if m:
        month = _MONTH_MAP.get(m.group(1).lower())
        if month:
            return f"{m.group(2)}-{month}-01", None, None

    # Try "since <Year>": "since 2024"
    m = re.match(r"(?:since|ab|seit)\s+(\d{4})", lower)
    if m:
        return f"{m.group(1)}-01-01", None, None

    # Try "until <Month> <Year>": "until March 2025"
    m = re.match(r"(?:until|bis|till)\s+(\w+)\s+(\d{4})", lower)
    if m:
        month = _MONTH_MAP.get(m.group(1).lower())
        if month:
            return None, f"{m.group(2)}-{month}-01", None

    # Try "until <Year>": "until 2025"
    m = re.match(r"(?:until|bis|till)\s+(\d{4})", lower)
    if m:
        return None, f"{m.group(1)}-12-31", None

    # Try "as of <Month> <Year>": "as of Q3 2024"
    m = re.match(r"(?:as of|stand)\s+(\w+)\s+(\d{4})", lower)
    if m:
        month = _MONTH_MAP.get(m.group(1).lower())
        if month:
            return f"{m.group(2)}-{month}-01", None, None

    # Try quarter: "Q3 2024", "Q1 2020"
    m = re.match(r"q(\d)\s*(\d{4})", lower)
    if m:
        q = m.group(1)
        year = m.group(2)
        start_month = _QUARTER_START.get(q)
        end_month = _QUARTER_END.get(q)
        if start_month and end_month:
            # Compute last day of end month (approximate)
            last_day = "30" if end_month in ("06", "09") else "31"
            if end_month == "03":
                last_day = "31"
            return f"{year}-{start_month}-01", f"{year}-{end_month}-{last_day}", None

    # Try "<Month> <Year>": "January 2024", "März 2025"
    m = re.match(r"(\w+)\s+(\d{4})", lower)
    if m:
        month = _MONTH_MAP.get(m.group(1).lower())
        if month:
            return f"{m.group(2)}-{month}-01", None, None

    # Try bare year: "2024"
    m = re.match(r"^(\d{4})$", scope.strip())
    if m:
        return f"{m.group(1)}-01-01", f"{m.group(1)}-12-31", None

    # Try ISO-8601 date: "2024-01-15"
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", scope.strip())
    if m:
        return m.group(1), None, None

    # Could not resolve -- return as qualifier
    return None, None, scope
