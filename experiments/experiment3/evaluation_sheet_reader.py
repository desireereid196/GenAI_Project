import pandas as pd
import os
import re
import numpy as np
import logging
from typing import Dict, List, Tuple
from tqdm import tqdm
from vtt.evaluation.metrics import ensure_nltk_resources

# Directly import functions needed for per-pair calculation
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
import bert_score

# Ensure NLTK resources are downloaded
ensure_nltk_resources()

# Configure script-specific logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    # Get the directory of this script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Current script directory: {current_script_dir}")

    # Adjust this path to where your XLSX file is located.
    # Assuming the XLSX is in the same directory as the script.
    evaluation_file_path = os.path.join(
        current_script_dir, "human_evaluation_sheet.xlsx"
    )

    # Call the updated function to read the sheet and compute per-pair metrics
    combined_df = read_evaluation_sheet_and_compute_metrics(evaluation_file_path)

    if not combined_df.empty:
        logger.info("\n--- Combined Evaluation DataFrame (First 5 Rows) ---")
        logger.info(f"Shape: {combined_df.shape}")
        logger.info(f"Columns: {combined_df.columns.tolist()}")
        logger.info("\nData Types:")
        logger.info(combined_df.dtypes)
        logger.info("\nFirst 5 Rows:")
        logger.info(combined_df.head())

        # The human ratings might be empty if the provided sheet is indeed 'blank'
        human_rating_cols = ["Adequacy (1-5)", "Fluency (1-5)", "Overall Quality (1-5)"]
        existing_human_cols = [
            col
            for col in human_rating_cols
            if col in combined_df.columns and combined_df[col].notna().any()
        ]
        if existing_human_cols:
            logger.info("\nDescriptive Statistics for Human Ratings:")
            logger.info(combined_df[existing_human_cols].describe())
        else:
            logger.info(
                "\nNo human rating data found or all are NaN in the loaded sheet."
            )

        # Print descriptive statistics for the newly added automatic metric columns
        automatic_metric_cols = [
            "BLEU-1",
            "BLEU-2",
            "BLEU-3",
            "BLEU-4",
            "METEOR",
            "BERTScore_F1",
            "BERTScore_P",
            "BERTScore_R",
        ]
        existing_auto_cols = [
            col for col in automatic_metric_cols if col in combined_df.columns
        ]
        if existing_auto_cols:
            logger.info("\nDescriptive Statistics for Per-Pair Automatic Metrics:")
            logger.info(combined_df[existing_auto_cols].describe())
        else:
            logger.warning(
                "\nNo automatic metric columns found in the dataframe to describe."
            )

    else:
        logger.error("\nNo evaluation data was loaded.")


def read_human_evaluation_sheets(directory_path: str) -> pd.DataFrame:
    """
    Reads all human evaluation Excel (.xlsx, .xls) files from a specified directory
    into a single pandas DataFrame.

    This function is kept for compatibility with the original script's `main`
    but is not directly used by `read_evaluation_sheet_and_compute_metrics`.
    If you were to combine multiple sheets, you'd adapt this function.
    """
    all_dataframes = []
    expected_headers = [
        "Image Filename",
        "Ground Truth Caption",
        "Generated Caption (Model)",
        "Adequacy (1-5)",
        "Fluency (1-5)",
        "Overall Quality (1-5)",
        "Comments",
        "Generation Method",
    ]

    if not os.path.isdir(directory_path):
        logger.error(f"Error: Directory not found at {directory_path}")
        return pd.DataFrame()

    logger.info(f"Scanning directory: {directory_path} for Excel files...")
    for filename in os.listdir(directory_path):
        if filename.endswith((".xlsx", ".xls")):
            filepath = os.path.join(directory_path, filename)
            try:
                # Read the Excel file. Assuming the data starts from the first sheet (0).
                # No nrows specified here, reads full sheet as per original function.
                df = pd.read_excel(filepath, sheet_name=0, nrows=50)

                if not all(col in df.columns for col in expected_headers):
                    logger.warning(
                        f"Warning: File {filename} is missing some expected columns."
                    )
                    missing_cols = [
                        col for col in expected_headers if col not in df.columns
                    ]
                    logger.warning(f"   Missing: {missing_cols}")

                if "Image" in df.columns:
                    df = df.drop(columns=["Image"])

                df["source_file"] = filename
                all_dataframes.append(df)
                logger.info(f"Successfully read: {filename}")

            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")

    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        rating_cols = ["Adequacy (1-5)", "Fluency (1-5)", "Overall Quality (1-5)"]
        for col in rating_cols:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")
        return combined_df
    else:
        logger.info(f"No suitable Excel files found in {directory_path}")
        return pd.DataFrame()


def read_evaluation_sheet_and_compute_metrics(file_path: str) -> pd.DataFrame:
    """
    Reads a single human evaluation XLSX file, computes automatic metrics for each
    (ground truth, generated caption) pair, and adds them as new columns to the DataFrame.

    Args:
        file_path (str): The path to the XLSX file.

    Returns:
        pd.DataFrame: The DataFrame with the evaluation data and computed per-pair metrics.
                      Returns an empty DataFrame if the file cannot be read or processed.
    """
    if not os.path.exists(file_path):
        logger.error(f"Error: File not found at {file_path}")
        return pd.DataFrame()

    logger.info(f"Reading evaluation sheet: {file_path}")
    try:
        # Read the XLSX file
        df = pd.read_excel(file_path, sheet_name=0)

        # Drop unnecessary columns for automatic metrics
        df = df.drop(
            columns=[
                "Image",
                "Adequacy (1-5)",
                "Fluency (1-5)",
                "Overall Quality (1-5)",
                "Comments",
            ]
        )

        # Validate essential columns
        required_cols = ["Ground Truth Caption", "Generated Caption (Model)"]
        if not all(col in df.columns for col in required_cols):
            logger.error(
                f"Error: Missing required columns in {file_path}. Expected: {required_cols}"
            )
            return pd.DataFrame()

        # Compute automatic metrics for each pair
        df_with_metrics = compute_per_pair_automatic_metrics(df)

        return df_with_metrics

    except Exception as e:
        logger.error(f"Error reading or processing {file_path}: {e}", exc_info=True)
        return pd.DataFrame()


def compute_per_pair_automatic_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes BLEU-1, -2, -3, -4, METEOR, and BERTScore for each pair of
    Ground Truth and Generated Captions and adds them as new columns to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'Ground Truth Caption' and 'Generated Caption (Model)' columns.

    Returns:
        pd.DataFrame: The original DataFrame with added columns for each computed metric.
    """
    # Initialize lists to store scores
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []
    meteor_scores = []
    bert_f1_scores = []
    bert_p_scores = []
    bert_r_scores = []

    # Using smoothing for BLEU scores
    smooth = SmoothingFunction().method1

    # Prepare lists for batch BERTScore calculation
    bert_references_batch = []
    bert_candidates_batch = []
    bert_original_indices = []

    # For progress bar

    logger.info("Calculating per-pair BLEU and METEOR scores...")
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        ground_truth = (
            str(row["Ground Truth Caption"]).strip()
            if pd.notna(row["Ground Truth Caption"])
            else ""
        )
        generated = (
            str(row["Generated Caption (Model)"]).strip()
            if pd.notna(row["Generated Caption (Model)"])
            else ""
        )

        if ground_truth and generated:
            # Tokenize captions
            # BLEU expects a list of list of tokens for references (even if only one reference)
            bleu_ref_tokens_list = [ground_truth.split()]  # List of lists of tokens

            # METEOR's meteor_score function expects the 'references' argument to be a list
            # of *single* pre-tokenized reference strings (list of tokens).
            meteor_ref_tokens = (
                ground_truth.split()
            )  # This is a list of tokens, e.g., ['hello', 'world']
            hypothesis_tokens = generated.split()  # This is also a list of tokens

            # --- Robustness: Filter out empty strings from token lists ---
            # This handles cases like "  ".split() -> [] or "word ".split() -> ['word', '']
            bleu_ref_tokens_list = [
                [token for token in ref if token] for ref in bleu_ref_tokens_list
            ]
            meteor_ref_tokens = [token for token in meteor_ref_tokens if token]
            hypothesis_tokens = [token for token in hypothesis_tokens if token]

            # Ensure token lists are not empty after filtering before computing metrics
            if (
                not bleu_ref_tokens_list[0] or not hypothesis_tokens
            ):  # Check first reference in BLEU list
                # If either is empty after cleaning, skip metric calculation for this pair
                bleu_1_scores.append(np.nan)
                bleu_2_scores.append(np.nan)
                bleu_3_scores.append(np.nan)
                bleu_4_scores.append(np.nan)
                meteor_scores.append(np.nan)
                # BERTScore will be handled by not adding to batch
                continue  # Skip to next row

            # BLEU
            bleu_1_scores.append(
                sentence_bleu(
                    bleu_ref_tokens_list,
                    hypothesis_tokens,
                    weights=(1, 0, 0, 0),
                    smoothing_function=smooth,
                )
            )
            bleu_2_scores.append(
                sentence_bleu(
                    bleu_ref_tokens_list,
                    hypothesis_tokens,
                    weights=(0.5, 0.5, 0, 0),
                    smoothing_function=smooth,
                )
            )
            bleu_3_scores.append(
                sentence_bleu(
                    bleu_ref_tokens_list,
                    hypothesis_tokens,
                    weights=(0.33, 0.33, 0.33, 0),
                    smoothing_function=smooth,
                )
            )
            bleu_4_scores.append(
                sentence_bleu(
                    bleu_ref_tokens_list,
                    hypothesis_tokens,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smooth,
                )
            )

            # METEOR
            if meteor_ref_tokens and hypothesis_tokens:
                try:
                    # Pass a list containing the single tokenized reference
                    meteor_scores.append(
                        meteor_score([meteor_ref_tokens], hypothesis_tokens)
                    )
                except TypeError as e:
                    logger.error(
                        f"METEOR calculation TypeError for Ground Truth: '{ground_truth}', Generated: '{generated}'"
                    )
                    logger.error(
                        f"  meteor_ref_tokens (passed): {meteor_ref_tokens}, type: {type(meteor_ref_tokens)}, elements type: {[type(t) for t in meteor_ref_tokens] if isinstance(meteor_ref_tokens, list) else 'N/A'}"
                    )
                    logger.error(
                        f"  hypothesis_tokens: {hypothesis_tokens}, type: {type(hypothesis_tokens)}, elements type: {[type(t) for t in hypothesis_tokens] if isinstance(hypothesis_tokens, list) else 'N/A'}"
                    )
                    logger.error(f"  Error: {e}")
                    meteor_scores.append(np.nan)
                except Exception as e:  # Catch any other unexpected errors
                    logger.error(
                        f"Unexpected error in METEOR calculation for Ground Truth: '{ground_truth}', Generated: '{generated}'"
                    )
                    logger.error(f"  Error: {e}")
                    meteor_scores.append(np.nan)
            else:
                meteor_scores.append(np.nan)

            # Store for batch BERTScore
            bert_references_batch.append(ground_truth)
            bert_candidates_batch.append(generated)
            bert_original_indices.append(idx)  # Store original index for mapping back
        else:
            # Append NaN if captions are missing or empty in the original DataFrame row
            bleu_1_scores.append(np.nan)
            bleu_2_scores.append(np.nan)
            bleu_3_scores.append(np.nan)
            bleu_4_scores.append(np.nan)
            meteor_scores.append(np.nan)
            # BERTScore will be filled after batch processing, NaNs for these rows will be handled below

    # Add BLEU and METEOR scores to DataFrame
    df["BLEU-1"] = bleu_1_scores
    df["BLEU-2"] = bleu_2_scores
    df["BLEU-3"] = bleu_3_scores
    df["BLEU-4"] = bleu_4_scores
    df["METEOR"] = meteor_scores

    # Calculate BERTScore in batch for efficiency
    logger.info("Calculating per-pair BERTScore (this may take a while)...")
    if bert_references_batch and bert_candidates_batch:
        # P, R, F1 are tensors, one value per pair in the batch
        P, R, F1 = bert_score.score(
            bert_candidates_batch, bert_references_batch, lang="en", verbose=False
        )

        # Initialize full columns with NaN
        df["BERTScore_P"] = np.nan
        df["BERTScore_R"] = np.nan
        df["BERTScore_F1"] = np.nan

        # Map batch results back to DataFrame based on original indices
        for i, original_idx in enumerate(bert_original_indices):
            df.loc[original_idx, "BERTScore_P"] = P[i].item()
            df.loc[original_idx, "BERTScore_R"] = R[i].item()
            df.loc[original_idx, "BERTScore_F1"] = F1[i].item()
    else:
        logger.warning("No valid caption pairs for BERTScore calculation.")
        df["BERTScore_P"] = np.nan
        df["BERTScore_R"] = np.nan
        df["BERTScore_F1"] = np.nan

    return df


if __name__ == "__main__":
    main()
