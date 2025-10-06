# tools/file_operations.py
"""File operations utilities (ODT reading and Excel export)."""

import pandas as pd
from typing import Dict, Any
import tempfile

# odfpy imports
from odf.opendocument import load as odf_load
from odf.table import Table, TableRow, TableCell
from odf.text import P


def read_odt_tables(uploaded_file) -> Dict[str, pd.DataFrame]:
    """Read an ODT document and extract tables as DataFrames.

    Args:
        uploaded_file: File-like with read() returning bytes.

    Returns:
        Mapping of table names to DataFrames. If a table has no name, a default
        name like 'table_1' is used.
    """
    # Read bytes without consuming the uploaded_file for other uses
    data = uploaded_file.read()
    # odfpy expects a path, so write to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.odt') as tmp:
        tmp.write(data)
        tmp.flush()
        doc = odf_load(tmp.name)

    tables = doc.getElementsByType(Table)
    result: Dict[str, pd.DataFrame] = {}
    for idx, tbl in enumerate(tables, start=1):
        rows = []
        for row in tbl.getElementsByType(TableRow):
            cells_text = []
            for cell in row.getElementsByType(TableCell):
                # Concatenate all paragraphs within the cell
                paragraphs = cell.getElementsByType(P)
                text_content = "\n".join([
                    str(p) if isinstance(p, str) else ''.join([n.data for n in p.childNodes if hasattr(n, 'data')])
                    for p in paragraphs
                ])
                cells_text.append(text_content)
            # Only append non-empty rows
            if any(cells_text):
                rows.append(cells_text)
        if rows:
            # Normalize row lengths
            max_len = max(len(r) for r in rows)
            rows = [r + [None] * (max_len - len(r)) for r in rows]
            df = pd.DataFrame(rows)
            # Try to promote first row to header if it looks like column names
            if not df.empty:
                first_row = df.iloc[0].tolist()
                def is_number_like(x):
                    try:
                        float(str(x).replace(',', '.'))
                        return True
                    except Exception:
                        return False
                non_numeric = [v for v in first_row if isinstance(v, str) and not is_number_like(v)]
                unique_ratio = len(set(first_row)) / max(1, len(first_row))
                if len(non_numeric) >= max(1, int(0.6 * len(first_row))) and unique_ratio > 0.8:
                    df.columns = first_row
                    df = df.iloc[1:].reset_index(drop=True)
            name = tbl.getAttribute('name') or f'table_{idx}'
            result[name] = df
    return result


def export_to_excel(df: pd.DataFrame, filename: str = "export.xlsx", sheet_name: str = "Data") -> bytes:
    """Export DataFrame to Excel with formatting and return bytes.

    Falls back to default engine or CSV if Excel engines are unavailable.
    """
    from io import BytesIO
    import pandas as pd

    output = BytesIO()
    try:
        # Try xlsxwriter first
        import xlsxwriter  # noqa: F401
        engine = 'xlsxwriter'
    except Exception:
        # Let pandas choose available engine; if none, fallback to CSV
        engine = None

    try:
        if engine:
            with pd.ExcelWriter(output, engine=engine) as writer:
                df.to_excel(writer, index=False, sheet_name=sheet_name)
                # Best-effort formatting
                try:
                    worksheet = writer.sheets[sheet_name]
                    for i, col in enumerate(df.columns):
                        width = max(10, min(40, int(df[col].astype(str).str.len().mean()) + 5))
                        worksheet.set_column(i, i, width)
                except Exception:
                    pass
        else:
            with pd.ExcelWriter(output) as writer:
                df.to_excel(writer, index=False, sheet_name=sheet_name)
        return output.getvalue()
    except Exception:
        # Fallback to CSV bytes
        return df.to_csv(index=False).encode()


def export_analysis_results(results: Dict[str, Any], filename: str = "analysis_results.xlsx") -> bytes:
    """Export analysis results (dict of DataFrames or dicts) to Excel and return bytes.

    Falls back to CSV of concatenated sheets if Excel engines are unavailable.
    """
    from io import BytesIO
    import pandas as pd

    output = BytesIO()
    try:
        import xlsxwriter  # noqa: F401
        engine = 'xlsxwriter'
    except Exception:
        engine = None

    try:
        if engine:
            with pd.ExcelWriter(output, engine=engine) as writer:
                for key, val in results.items():
                    sheet = str(key)[:31]
                    if isinstance(val, pd.DataFrame):
                        val.to_excel(writer, index=False, sheet_name=sheet)
                    else:
                        try:
                            df = pd.DataFrame(val)
                        except Exception:
                            df = pd.DataFrame({'value': [str(val)]})
                        df.to_excel(writer, index=False, sheet_name=sheet)
            return output.getvalue()
        else:
            # Fallback: write a simple CSV of the first sheet-like item
            # Prefer first DataFrame, else serialize dict to DataFrame
            for key, val in results.items():
                if isinstance(val, pd.DataFrame):
                    return val.to_csv(index=False).encode()
            # If no DataFrame found, serialize entire dict
            return pd.DataFrame(results).to_csv(index=False).encode()
    except Exception:
        # As a last resort, bytes of string representation
        return str(results).encode()
