"""
Script: evaluation_sheet_reader.py
Description: Reads human evaluation data from a single Excel (.xlsx) file
             into a pandas DataFrame for analysis and comparison of decoding strategies
             (e.g., Greedy vs Beam) based on error annotations.
"""

import pandas as pd
import os


def main():
    # Get the directory of this script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Current script directory:", current_script_dir)

    excel_filename = "error_taxonomy_sheet.xlsx" 
    excel_filepath = os.path.join(current_script_dir, excel_filename)

    evaluation_dataframe = read_single_evaluation_sheet(excel_filepath)

    if not evaluation_dataframe.empty:
        print("\n--- Evaluation DataFrame Loaded ---")
        print("Shape:", evaluation_dataframe.shape)
        print("\nColumns:", evaluation_dataframe.columns.tolist())
        print("\nFirst 5 Rows:")
        print(evaluation_dataframe.head())

        # Descriptive stats for all error categories
        error_cols = [
            "Hallucination", "Omission", "Ambiguity", "Grammatical Error",
            "Repetition", "Wrong Action", "Fine-Grained Error", "Correct (No Error)"
        ]
        available_error_cols = [col for col in error_cols if col in evaluation_dataframe.columns]

        print("\nDescriptive Statistics for Error Annotations:")
        print(evaluation_dataframe[available_error_cols].describe())

    else:
        print("\nNo evaluation data was loaded.")


def read_single_evaluation_sheet(file_path: str) -> pd.DataFrame:
    """
    Reads a single Excel evaluation file with multiple rows per image (e.g., per Search Type).

    Args:
        file_path (str): Full path to the Excel file.

    Returns:
        pd.DataFrame: DataFrame with parsed evaluation data.
    """
    expected_headers = [
        "Image Filename",
        "Ground Truth Caption",
        "Search Type",
        "Generated Caption",
        "Hallucination",
        "Omission",
        "Ambiguity",
        "Grammatical Error",
        "Repetition",
        "Wrong Action",
        "Fine-Grained Error",
        "Correct (No Error)"
    ]

    if not os.path.isfile(file_path):
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_excel(file_path, sheet_name=0)

        # Fill down merged cells (e.g., Image Filename, Ground Truth Caption)
        df[['Image Filename', 'Ground Truth Caption']] = df[['Image Filename', 'Ground Truth Caption']].ffill()

        # Warn if columns are missing
        missing_cols = [col for col in expected_headers if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing expected columns: {missing_cols}")

        # Drop image column if it exists (often contains embedded images)
        if "Image" in df.columns:
            df = df.drop(columns=["Image"])

        # Add source file column for traceability
        df["source_file"] = os.path.basename(file_path)

        # Convert Yes/No error columns to binary (1 = Yes, 0 = No)
        error_cols = [
            "Hallucination", "Omission", "Ambiguity", "Grammatical Error",
            "Repetition", "Wrong Action", "Fine-Grained Error", "Correct (No Error)"
        ]
        for col in error_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower().map(
                    {"yes": 1, "no": 0}
                ).astype("Int64")

        return df

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    main()
