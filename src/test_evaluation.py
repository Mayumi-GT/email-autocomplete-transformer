import os
# This was added to try and prevent a specific 'semaphore' warning in both mac and CoLab.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# src/test_evaluation.py

import pandas as pd
import time
from tqdm import tqdm  # For visual progress bars

"""
This file manages the entire workflow from start to finish: loading the raw data, 
preprocessing it, splitting it, preparing the N-gram and Transformer models, 
generating the specific test cases (prompt/continuation pairs), running each model 
on these test cases, calculating performance metrics using `evaluation.py`, 
and reporting the results.
"""

# Import revised functions from other files
from src.data_preprocessing import (
    load_full_dataset,
    preprocess_data,
    split_data,
    create_prompt_continuation_pairs
)
from src.model import (
    setup_transformer_model,
    transformer_autocomplete,
    # fine_tune_transformer 
)
from src.baseline_ngram import (
    train_ngram_model,
    ngram_autocomplete,
    train_char_ngram_model,
    char_ngram_autocomplete,
    load_ngram_model
)
from src.evaluation import (
    compute_next_word_accuracy,
    compute_next_char_accuracy,
    calculate_keystroke_savings_rate
)


DATA_FILE_PATH = "/content/drive/MyDrive/project/emails.csv"
# Models to switch: chose between the two below by commenint gone out.
# TRANSFORMER_MODEL_ID = "/content/drive/MyDrive/project/fine_tuned_distilgpt2_subset_test" 
TRANSFORMER_MODEL_ID = "distilgpt2"  # "distilgpt2" ("gpt2" was too slow/ terminated unexpectedly) 

# Paths to save/ load trained N-gram models (apparently saves retraining time)
WORD_NGRAM_MODEL_PATH = "word_ngram_model.pkl"
CHAR_NGRAM_MODEL_PATH = "char_ngram_model.pkl"
# Evaluation parameters
N_GRAM_ORDER = 3  # Order for N-gram
NUM_TEST_PAIRS = 100  # Number of prompt/continuation pairs to evaluate on (from test set) #100 made it run  >30min so changing to lower to check if the code works.
TRANSFORMER_MAX_NEW_TOKENS = 30  # Max tokens for Transformer
NGRAM_WORD_SUGGEST_LEN = 5  # Number of words for N-gram word
NGRAM_CHAR_SUGGEST_LEN = 50  # Number of chars for N-gram char
KSR_MIN_MATCH_LEN = 3  # Min match length for KSR calculation



def run_evaluation():
    print("--- Starting Evaluation Pipeline ---")

    print("\n[1. Loading and Preprocessing Data]")
    full_df = load_full_dataset(DATA_FILE_PATH)
    if full_df is None or full_df.empty:
        print("Failed to load data. Exiting.")
        return

    processed_df = preprocess_data(full_df, text_col='message')
    if processed_df.empty:
        print("Preprocessing resulted in empty dataframe. Exiting.")
        return

    print("\n[2. Splitting Data]")
    train_df, val_df, test_df = split_data(processed_df)
    if test_df.empty:
        print("Test set is empty. Cannot proceed with evaluation.")
        return

    train_texts = train_df['clean_text'].tolist()
    test_texts = test_df['clean_text'].tolist()

    print("\n[3. Creating Evaluation Pairs from Test Set]")
    # Create pairs using a subset of test texts if test_df is very large
    evaluation_pairs = create_prompt_continuation_pairs(test_texts)
    if not evaluation_pairs:
        print("Could not create evaluation pairs. Check text lengths and pair creation logic.")
        return

    # Limit the number of pairs for evaluation as needed
    if len(evaluation_pairs) > NUM_TEST_PAIRS:
        print(f"Sampling {NUM_TEST_PAIRS} pairs for evaluation.")
        # Seed random sampling for reproducibility if needed
        import random
        random.seed(42)
        evaluation_pairs = random.sample(evaluation_pairs, NUM_TEST_PAIRS)
    else:
        print(f"Using {len(evaluation_pairs)} pairs for evaluation.")

    print("\n[4. Training/Loading N-gram Models]")

    # n-gram
    NGRAM_TRAIN_SAMPLE_SIZE = 50000  # this setting worked, # of training emails to use for N-grams
    if len(train_texts) > NGRAM_TRAIN_SAMPLE_SIZE:
        print(f"Sampling {NGRAM_TRAIN_SAMPLE_SIZE} texts for N-gram training (from {len(train_texts)} total).")
        import random
        random.seed(42)  # for reproducibility
        ngram_train_texts_sample = random.sample(train_texts, NGRAM_TRAIN_SAMPLE_SIZE)
    else:
        print("Using full training set for N-gram training.")
        ngram_train_texts_sample = train_texts

    # Word N-gram
    word_ngram_model = None
    if os.path.exists(WORD_NGRAM_MODEL_PATH):
        print(f"Loading existing word n-gram model from {WORD_NGRAM_MODEL_PATH}")
        word_ngram_model = load_ngram_model(WORD_NGRAM_MODEL_PATH)
    if word_ngram_model is None and ngram_train_texts_sample:  
        print("Training word n-gram model on sample...")
        start_time = time.time()
        word_ngram_model = train_ngram_model(ngram_train_texts_sample, n=N_GRAM_ORDER,
                                             model_save_path=WORD_NGRAM_MODEL_PATH)
        print(f"Word n-gram training took {time.time() - start_time:.2f} seconds.")

    # Character N-gram
    char_ngram_model = None
    if os.path.exists(CHAR_NGRAM_MODEL_PATH):
        print(f"Loading existing character n-gram model from {CHAR_NGRAM_MODEL_PATH}")
        char_ngram_model = load_ngram_model(CHAR_NGRAM_MODEL_PATH)
    # Use the SAMPLE for training if model wasn't loaded
    if char_ngram_model is None and ngram_train_texts_sample: 
        print("Training character n-gram model on sample...")
        start_time = time.time()
        char_ngram_model = train_char_ngram_model(ngram_train_texts_sample, n=N_GRAM_ORDER,
                                                  model_save_path=CHAR_NGRAM_MODEL_PATH)
        print(f"Character n-gram training took {time.time() - start_time:.2f} seconds.")

    print(f"\n[5. Loading Transformer Model: {TRANSFORMER_MODEL_ID}]")
    transformer_tokenizer, transformer_model, transformer_device = setup_transformer_model(TRANSFORMER_MODEL_ID)

    if not transformer_tokenizer or not transformer_model:
        print("Failed to load Transformer model. Cannot proceed with its evaluation.")
        # Optionally decide whether to continue with N-grams only or exit
        # return

    # test block used for debuggnig to check it is starting/ working:
    if transformer_tokenizer and transformer_model:
        print("\n[5b. Testing Basic Transformer Inference]")
        try:
            test_prompt = "Hello world"
            print(f"--- Running basic inference test with prompt: '{test_prompt}' ---")
            test_suggestion = transformer_autocomplete(
                transformer_tokenizer,
                transformer_model,
                test_prompt,
                max_new_tokens=5,
                device=transformer_model.device
            )
            print(f"--- Basic inference test successful. Suggestion: '{test_suggestion}' ---")
        except Exception as e:
            print(f"--- Basic inference test FAILED with Python error: {e} ---")
            import traceback
            traceback.print_exc() 
        print("[5c. Basic Inference Test Complete]")
    else:
        print("[5b. Skipping basic inference test - model not loaded]")


    print(f"\n[6. Running Evaluation on {len(evaluation_pairs)} Pairs]")
    results = []

    for i, (prompt, ground_truth) in enumerate(tqdm(evaluation_pairs, desc="Evaluating Pairs")):
        result_row = {'prompt': prompt, 'ground_truth': ground_truth}

        # --- Transformer Prediction ---
        if transformer_tokenizer and transformer_model:
            tf_suggestion = None  # Initialize
            tf_time = 0
            tf_word_acc = 0
            tf_char_acc = 0
            tf_ksr = 0.0

            if i == 0: print("Attempting transformer_autocomplete for first pair...")
            try:
                start_time = time.time()
                tf_suggestion = transformer_autocomplete(
                    transformer_tokenizer,
                    transformer_model,
                    prompt,
                    max_new_tokens=TRANSFORMER_MAX_NEW_TOKENS,
                    device=transformer_model.device
                )
                tf_time = time.time() - start_time
                if i == 0: print(
                    f"transformer_autocomplete succeeded for first pair. Suggestion: '{tf_suggestion[:50]}...'")

                # Calculate metrics only if suggestion was successful
                tf_word_acc = compute_next_word_accuracy(ground_truth, tf_suggestion)
                tf_char_acc = compute_next_char_accuracy(ground_truth, tf_suggestion)
                tf_ksr = calculate_keystroke_savings_rate(ground_truth, tf_suggestion, KSR_MIN_MATCH_LEN)

            except Exception as e:
                if i == 0:
                    print(f"!!! transformer_autocomplete FAILED for first pair with Python error: {e} !!!")
                    import traceback
                    traceback.print_exc()  
               
                tf_suggestion = "[ERROR]"
                tf_time = time.time() - start_time  # Time until error occurred
                # Metrics remain 0 / 0.0

            # Store results
            result_row['transformer_suggestion'] = tf_suggestion
            result_row['transformer_time'] = tf_time
            result_row['transformer_word_acc'] = tf_word_acc
            result_row['transformer_char_acc'] = tf_char_acc
            result_row['transformer_ksr'] = tf_ksr  

        else:
            result_row.update({k: None for k in ['transformer_suggestion', 'transformer_time', 'transformer_word_acc',
                                                 'transformer_char_acc', 'transformer_ksr']})

        # Word N-gram
        if word_ngram_model:
            start_time = time.time()
            wn_suggestion = ngram_autocomplete(word_ngram_model, prompt.lower(), num_words=NGRAM_WORD_SUGGEST_LEN)
            wn_time = time.time() - start_time
            result_row['word_ngram_suggestion'] = wn_suggestion
            result_row['word_ngram_time'] = wn_time
            result_row['word_ngram_word_acc'] = compute_next_word_accuracy(ground_truth, wn_suggestion)
            result_row['word_ngram_char_acc'] = compute_next_char_accuracy(ground_truth,
                                                                           wn_suggestion)  # Can still calc char acc
            result_row['word_ngram_ksr'] = calculate_keystroke_savings_rate(ground_truth, wn_suggestion,
                                                                            KSR_MIN_MATCH_LEN)
        else:
            result_row.update({k: None for k in ['word_ngram_suggestion', 'word_ngram_time', 'word_ngram_word_acc',
                                                 'word_ngram_char_acc', 'word_ngram_ksr']})

        # Char N-gram
        if char_ngram_model:
            start_time = time.time()
            cn_suggestion = char_ngram_autocomplete(char_ngram_model, prompt.lower(), num_chars=NGRAM_CHAR_SUGGEST_LEN)
            cn_time = time.time() - start_time
            result_row['char_ngram_suggestion'] = cn_suggestion
            result_row['char_ngram_time'] = cn_time
            # Word accuracy doesn't make sense for char model unless we define rules
            result_row['char_ngram_word_acc'] = 0.0
            result_row['char_ngram_char_acc'] = compute_next_char_accuracy(ground_truth, cn_suggestion)
            result_row['char_ngram_ksr'] = calculate_keystroke_savings_rate(ground_truth, cn_suggestion,
                                                                            KSR_MIN_MATCH_LEN)
        else:
            result_row.update({k: None for k in ['char_ngram_suggestion', 'char_ngram_time', 'char_ngram_word_acc',
                                                 'char_ngram_char_acc', 'char_ngram_ksr']})

        results.append(result_row)

  
    print("\n[7. Aggregating Results]")
    results_df = pd.DataFrame(results)

    # Save detailed results to CSV (to check
    results_df.to_csv("evaluation_results_detailed.csv", index=False)
    print("Detailed results saved to evaluation_results_detailed.csv")

    # Calculate average metrics, ignoring None values from failed models/predictions
    avg_metrics = {}
    metric_cols = [col for col in results_df.columns if '_acc' in col or '_ksr' in col or '_time' in col]

    for col in metric_cols:
        # Check if column exists and has non-NA values before calculating mean
        if col in results_df and results_df[col].notna().any():
            avg_metrics[f'avg_{col}'] = results_df[col].mean(skipna=True)
        else:
            avg_metrics[f'avg_{col}'] = None  

    print("\n--- Evaluation Summary ---")
    print(f"Dataset: {DATA_FILE_PATH}")
    print(f"Transformer Model: {TRANSFORMER_MODEL_ID}")
    print(f"N-gram Order: {N_GRAM_ORDER}")
    print(f"Evaluation Pairs: {len(results_df)}")
    print("-" * 25)
    print("Average Metrics:")

    print("\nTransformer:")
    if avg_metrics.get('avg_transformer_time') is not None:
        print(f"  Avg. Inference Time: {avg_metrics['avg_transformer_time']:.4f} sec")
        print(f"  Avg. Next Word Acc:  {avg_metrics['avg_transformer_word_acc']:.4f}")
        print(f"  Avg. Next Char Acc:  {avg_metrics['avg_transformer_char_acc']:.4f}")
        print(f"  Avg. KSR:            {avg_metrics['avg_transformer_ksr']:.4f}")
    else:
        print("  (Not Evaluated)")

    print("\nWord N-gram:")
    if avg_metrics.get('avg_word_ngram_time') is not None:
        print(f"  Avg. Inference Time: {avg_metrics['avg_word_ngram_time']:.4f} sec")
        print(f"  Avg. Next Word Acc:  {avg_metrics['avg_word_ngram_word_acc']:.4f}")
        print(f"  Avg. Next Char Acc:  {avg_metrics['avg_word_ngram_char_acc']:.4f}")  # Note: Char acc on word model
        print(f"  Avg. KSR:            {avg_metrics['avg_word_ngram_ksr']:.4f}")
    else:
        print("  (Not Evaluated)")

    print("\nCharacter N-gram:")
    if avg_metrics.get('avg_char_ngram_time') is not None:
        print(f"  Avg. Inference Time: {avg_metrics['avg_char_ngram_time']:.4f} sec")
        print(f"  Avg. Next Char Acc:  {avg_metrics['avg_char_ngram_char_acc']:.4f}")
        print(f"  Avg. KSR:            {avg_metrics['avg_char_ngram_ksr']:.4f}")
    else:
        print("  (Not Evaluated)")

    print("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    run_evaluation()
