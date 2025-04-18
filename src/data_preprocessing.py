# src/data_preprocessing.py
import pandas as pd
import re # for regular expression
from sklearn.model_selection import train_test_split # this imports specific function from scikit-learn that automatically splits data into training and testing sets.

"""
Purpose:
This file is to take the raw email dataset (email.csv), clean up the contect,
then split into sets of training, validation, and tests to build and evaluate
machine learning models. This file also includes a function to create `prompt`
and `answer` pairs needed later for testing the autocomplete models

Functions:
1. load_full_dataset(file_path)
2. clean_text(text)
3. preprocess_data(df, text_col='message')
4. split_data(df, test_size=0.15, validation_size=0.15, random_state=42)
5. create_prompt_continuation_pairs(texts, min_prompt_words=5, max_prompt_words=25, min_continuation_words=3):

Reference:
- Jiawei Han, Micheline Kamber, Jian Pei. Data Mining: Concepts and Techniques, 3rd Edition. Morgan
Kaufmann, 2011. (CSPB 4502 Data Mining Textbook)
"""




def load_full_dataset(file_path):
    """
    Load the full dataset from CSV. Handles basic errors and cleaning.
    Args:
        file_path (str): The path to the CSV file (e.g., '/path/to/emails.csv').
    Returns:
        pandas.DataFrame: The loaded data as a DataFrame, or None if loading fails.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} rows from {file_path}")

        # 'message' is the main text column
        if 'message' not in df.columns: 
            raise ValueError("CSV file must contain a 'message' column.")

        # Handle potential missing values in the text column
        # Remove rows where the 'message' column has a missing value (NaN - Not a Number) inplace
        df.dropna(subset=['message'], inplace=True)

        # Ensure text column is string type
        df['message'] = df['message'].astype(str)  
        print(f"Dataset size after dropping NA in 'message': {len(df)} rows")

        # if everything works, return the dataframe.
        return df

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    except Exception as e:
        print(f"An error occurred during loading: {e}")
        return None


def clean_text(text):
    """
    Clean text by removing URLs, non-alphanumeric characters (optional),
    and extra whitespace. Convert to lowercase.
    *** this is one of places where there can be other modification/ cleaning options
    to have better accuracy and KSRs ***
    Args:
        text (str): The input text string.
    Returns:
        str: The cleaned text string.
    """
    if not isinstance(text, str):
        return ""  # Return empty if input is not text

    # convert all to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)  

    # Replace multiple spaces/newlines with one
    text = re.sub(r'\s+', ' ', text)  

    return text.strip() # remove any whitespace in the very beginning or the end.


def preprocess_data(df, text_col='message'):
    """
    Apply cleaning to the specified text column in the DataFrame.
    Adds a 'clean_text' column and removes rows that become empty after cleaning.
    Args:
        df (pandas.DataFrame): The input DataFrame.
        text_col (str): The name of the column containing the text to clean.
    Returns:
        pandas.DataFrame: The DataFrame with the added 'clean_text' column.
    """
    if text_col not in df.columns:
        print(f"Warning: Column '{text_col}' not found. Skipping preprocessing.")
        return df

    df['clean_text'] = df[text_col].apply(clean_text)
    # Remove rows where clean_text became empty after cleaning
    df = df[df['clean_text'] != '']
    print(f"Dataset size after cleaning and removing empty texts: {len(df)} rows")
    return df


def split_data(df, test_size=0.15, validation_size=0.15, random_state=42):
    """
    Split the DataFrame into training (70%), validation (15%), and test sets (15%).
    Args:
        df (DataFrame): The DataFrame to split (should have 'clean_text').
        test_size (float): Proportion for the test set.
        validation_size (float): Proportion for the validation set (from the remaining data).
        random_state (int): Random seed for reproducibility.
    Returns:
        tuple: (df_train, df_val, df_test)
    """
    if 'clean_text' not in df.columns:
        raise ValueError("DataFrame must have 'clean_text' column for splitting.")

    if len(df) < 3:
        print("Warning: Dataset too small for reliable train/val/test split.")
        return df, pd.DataFrame(), pd.DataFrame()  # Return original df as train, empty for val/test

    # Split off the test set first
    train_val_df, test_df = train_test_split(
        df, # DataFrame to split
        test_size=test_size,
        random_state=random_state
    )

    # calculate validation size relative to the remaining data
    relative_val_size = validation_size / (1.0 - test_size)

    # edge cases where the remaining dataset is too small for the relative validation split
    if len(train_val_df) < 2 or relative_val_size >= 1.0 or relative_val_size <= 0.0:
        print("Warning: Not enough data for validation split after test split. Assigning remaining to train.")
        train_df = train_val_df
        val_df = pd.DataFrame(columns=df.columns)
    else:
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=random_state
        )

    print(f"Data split: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df


def create_prompt_continuation_pairs(texts, min_prompt_words=5, max_prompt_words=25, min_continuation_words=3):
    """
    Creates (prompt, continuation) pairs from a list of texts for evaluation.
    Args:
        texts (list of str): A list of cleaned text documents.
        min_prompt_words (int): Minimum words in the generated prompt.
        max_prompt_words (int): Maximum words in the generated prompt.
        min_continuation_words (int): Minimum words required in the continuation part.
    Returns:
        list: A list of tuples, where each tuple is (prompt, continuation).
    """
    pairs = [] # initialize to hold generated pairs
    for text in texts:
        words = text.split()  # split where whitespace is
        if len(words) >= min_prompt_words + min_continuation_words:
            # Determine a split point within the allowed range
            split_point = min(max_prompt_words, len(words) - min_continuation_words)
            if split_point < min_prompt_words:
                continue  # Skip if can't even make the minimum prompt length

            prompt = " ".join(words[:split_point])
            continuation = " ".join(words[split_point:])

            # Ensure continuation meets minimum length
            if len(continuation.split()) >= min_continuation_words:
                pairs.append((prompt, continuation))

    print(f"Created {len(pairs)} prompt/continuation pairs.")
    return pairs

# only to test if above functions work,
if __name__ == "__main__":
    
    file_path = "/content/drive/MyDrive/project/emails.csv"

    full_df = load_full_dataset(file_path)

    if full_df is not None:
        processed_df = preprocess_data(full_df, text_col='message')

        if not processed_df.empty:
            train_df, val_df, test_df = split_data(processed_df)

            print("\n--- Train Set Sample ---")
            print(train_df.head())

            print("\n--- Validation Set Sample ---")
            print(val_df.head())

            print("\n--- Test Set Sample ---")
            print(test_df.head())

            test_texts = test_df['clean_text'].tolist()
            evaluation_pairs = create_prompt_continuation_pairs(test_texts)

            if evaluation_pairs:
                print("\n--- Sample Evaluation Pairs (Prompt, Continuation) ---")
                for i in range(min(5, len(evaluation_pairs))):
                    print(f"Prompt:       '{evaluation_pairs[i][0]}'")
                    print(f"Continuation: '{evaluation_pairs[i][1]}'\n")
            else:
                print("\nCould not generate evaluation pairs from the test set (texts might be too short).")
        else:
            print("Preprocessing resulted in an empty DataFrame.")
    else:
        print("Failed to load the dataset.")

