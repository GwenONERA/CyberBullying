#!/usr/bin/env python3
"""
Concatenate all Excel files under ./cyberbBullying into one Parquet file,
preserving all original cell values. Object-typed columns (where mixed
types typically live) are converted to pandas' nullable string dtype to
avoid parquet conversion errors while preserving every element.

"""

from pathlib import Path
import argparse
import sys
import pandas as pd
from pandas.api import types as ptypes

def read_excel_file(path: Path, sheet_name=0):
    try:
        # Let pandas choose engine (openpyxl/xlrd as available)
        return pd.read_excel(path, sheet_name=sheet_name, engine=None)
    except Exception as e:
        raise RuntimeError(f"Failed to read '{path}': {e}")

def sanitize_for_parquet(df: pd.DataFrame):
    """
    Modify df in-place to use parquet-friendly dtypes while preserving original
    textual representations for mixed-type columns.

    Strategy:
      - object dtype -> convert to pandas nullable string dtype (preserves content)
      - integer dtype with NA -> convert to pandas nullable Int64
      - boolean dtype with NA -> convert to pandas nullable boolean
      - datetime dtypes left alone
    Returns a list of columns that were converted to string.
    """
    converted_to_string = []
    for col in df.columns:
        dtype = df[col].dtype

        # If object, likely mixed types (e.g., some ints, some "15:35:30" strings).
        if dtype == "object":
            # Convert to pandas' nullable string dtype to preserve exact textual values
            df[col] = df[col].astype("string")
            converted_to_string.append(col)
            continue

        # If integer but contains NA, convert to nullable integer dtype
        if ptypes.is_integer_dtype(dtype):
            if df[col].isna().any():
                try:
                    df[col] = df[col].astype("Int64")
                except Exception:
                    # fallback: convert to string to be safe
                    df[col] = df[col].astype("string")
                    converted_to_string.append(col)
            # else leave as int64

        # If float that actually holds integers + NaNs you might want to keep as float
        # If boolean with NA -> use nullable boolean
        if ptypes.is_bool_dtype(dtype):
            if df[col].isna().any():
                try:
                    df[col] = df[col].astype("boolean")
                except Exception:
                    df[col] = df[col].astype("string")
                    converted_to_string.append(col)

        # datetime-like left as-is (pyarrow handles datetime64[ns])
        # category left as-is

    return converted_to_string

def main():
    parser = argparse.ArgumentParser(description="Combine XLSX files into a Parquet while preserving all cell values.")
    parser.add_argument("--input-dir", "-i", type=Path, default=Path("./cyberbBullying"), help="Directory with Excel files")
    parser.add_argument("--output-file", "-o", type=Path, default=Path("./cyberbBullying/combined_all.parquet"), help="Output Parquet path")
    parser.add_argument("--assert-count", type=int, default=None, help="Assert final row count equals this number")
    parser.add_argument("--no-schema-check", action="store_true", help="Disable strict column equality check")
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: '{input_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    excel_files = sorted([p for p in input_dir.glob("**/*") if p.suffix.lower() in {".xlsx", ".xls"}])
    if not excel_files:
        print("No Excel files found.", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(excel_files)} Excel files.")

    dfs = []
    ref_cols = None
    for p in excel_files:
        print(f"Reading {p.name} ...", end=" ")
        try:
            df = read_excel_file(p)
        except Exception as e:
            print(f"FAILED ({e})")
            sys.exit(3)
        print(f"ok ({len(df)} rows)")

        if ref_cols is None:
            ref_cols = list(df.columns)
        else:
            if not args.no_schema_check and list(df.columns) != ref_cols:
                print(f"Schema mismatch in file {p.name}", file=sys.stderr)
                print(f"Expected: {ref_cols}", file=sys.stderr)
                print(f"Found:    {list(df.columns)}", file=sys.stderr)
                sys.exit(4)

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined shape: {combined.shape}")
    print(f"Columns: {list(combined.columns)}")

    # Sanitize dtypes for reliable Parquet writing while preserving values
    converted = sanitize_for_parquet(combined)
    if converted:
        print(f"Converted {len(converted)} object/mixed columns to nullable string dtype (preserves all text values):")
        for c in converted:
            print(f"  - {c}")
    else:
        print("No object/mixed columns required conversion.")

    # Final sanity: ensure columns order and names preserved
    # (already preserved by concatenation and schema-check)
    # Create output dir
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write using pyarrow for best compatibility
    try:
        combined.to_parquet(args.output_file, index=False, engine="pyarrow")
    except Exception as e:
        print(f"Failed to write parquet: {e}", file=sys.stderr)
        # Try fallback: convert *all* columns to string to guarantee write (very safe)
        try:
            print("Attempting fallback: convert every column to string and retry...", file=sys.stderr)
            for col in combined.columns:
                combined[col] = combined[col].astype("string")
            combined.to_parquet(args.output_file, index=False, engine="pyarrow")
            print("Fallback succeeded (all columns written as strings).")
        except Exception as e2:
            print(f"Fallback also failed: {e2}", file=sys.stderr)
            sys.exit(5)

    print(f"Wrote Parquet to: {args.output_file}")

    if args.assert_count is not None:
        final_count = len(combined)
        if final_count != args.assert_count:
            print(f"Row count assertion failed: expected {args.assert_count}, got {final_count}.", file=sys.stderr)
            sys.exit(6)
        else:
            print(f"Row count assertion passed ({args.assert_count}).")

if __name__ == "__main__":
    main()
