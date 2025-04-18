# src/baseline_ngram.py
import nltk
from nltk.lm import Laplace # import N-gram model with Laplace smoothing
from nltk.lm.preprocessing import padded_everygram_pipeline # helper function
from nltk.tokenize import word_tokenize
import pickle  # To save and load trained models (python object)

"""
This file consists of "N-grams" models. This N-gram model predicts the next item,
word or character, based on previous few items by counging how often seqquences 
appears in the training text.

Functions:
1. train_ngram_model(train_texts, n=3, model_save_path=None)
2. ngram_autocomplete(model, prompt, num_words=5)
3. train_char_ngram_model(train_texts, n=5, model_save_path=None)
4. char_ngram_autocomplete(model, prompt, num_chars=20)
5. load_ngram_model(model_path)

Reference:
- CSPB 4830 Special Topic, Natural Language Processing, lecture and homework assignments
- https://www.nltk.org/bookLinks
- https://web.stanford.edu/~jurafsky/slp3/
"""


# NLTK download
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt')


def train_ngram_model(train_texts, n=3, model_save_path=None):
    """
    Train a baseline n-gram language model using Laplace smoothing 
    on provided training texts.

    Args:
        train_texts (list of str): List of texts from the training set.
        n (int): The order of the n-gram model.
        model_save_path (str, optional): Path to save the trained model. Defaults to None.

    Returns:
        nltk.lm.api.LanguageModel: The trained n-gram language model.
    """
    print(f"Training {n}-gram word model on {len(train_texts)} texts...")
    
    # Tokenize the corpus at the word-level
    tokenized_texts = [word_tokenize(text) for text in
                       train_texts]  

    # edege case
    if not any(tokenized_texts):
        print("Warning: No tokens found in training data for word n-gram model. Returning None.")
        return None

    # prepare data for model
    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_texts)

    model = Laplace(n)  
    model.fit(train_data, padded_sents)
    print(f"Finished training {n}-gram word model.")

    # Save the model if a path is provided
    if model_save_path:
        try:
            with open(model_save_path, 'wb') as f_out:
                pickle.dump(model, f_out)
            print(f"Word N-gram model saved to {model_save_path}")
        except Exception as e:
            print(f"Error saving word n-gram model: {e}")

    return model


def ngram_autocomplete(model, prompt, num_words=5):
    """
    Generate probable next words using the trained n-gram model.

    Args:
        model: A trained n-gram model.
        prompt (str): The starting text (should be lowercased).
        num_words (int): Number of words to predict.

    Returns:
        str: The predicted continuation (sequence of words). Returns empty string if model is None or cannot generate.
    """

    # edge case
    if model is None:
        return ""

    try:
        tokens = word_tokenize(prompt) 
        
        # The last 'n-1' words of the prompt.
        # 'model.order' gives 'n' (e.g., 3). So 'model.order - 1' is 2.
        # `tokens[-(n-1):]` gets the last n-1 elements from the list.
        # If n=1 (unigram), context is empty.
        context = tokens[-(model.order - 1):] if model.order > 1 else []

        generated_words = []
        for _ in range(num_words):
            # Use the last n-1 generated words as context for the next prediction
            current_context = (generated_words + tokens)[-(model.order - 1):] if model.order > 1 else []
            # edge case if context is empty or unknown
            try:
                # text_seed expects list of tokens
                next_word = model.generate(text_seed=current_context)
                # Stop if generate returns end-of-sequence token or similar marker if model uses one (Laplace doesn't explicitly)
                # Or if it repeats excessively (simple check)
                if not next_word or next_word == '</s>' or next_word in generated_words[-2:]:
                    break
                generated_words.append(next_word)
            except Exception as e:
                # print(f"N-gram generation error for context {current_context}: {e}") # Optional debug
                break  # Stop generation on error

        return ' '.join(generated_words)
    except Exception as e:
        print(f"Error during n-gram autocomplete: {e}")
        return ""


def train_char_ngram_model(train_texts, n=5, model_save_path=None):
    """
    Similar to words, the model learns probability of next character based on
    previous n-1 character

    Args:
        train_texts (list of str): List of texts from the training set.
        n (int): Order of the n-gram model, set as 5.
        model_save_path (str, optional): Path to save the trained model. Defaults to None.

    Returns:
        nltk.lm.api.LanguageModel: The trained character-level model.
    """
    print(f"Training {n}-gram character model on {len(train_texts)} texts...")
    tokenized_texts = [list(text) for text in train_texts]  # Assumes text is lowercased

    if not any(tokenized_texts):
        print("Warning: No characters found in training data for char n-gram model. Returning None.")
        return None

    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_texts)

    model = Laplace(n)
    model.fit(train_data, padded_sents)
    print(f"Finished training {n}-gram character model.")

    # Save the model
    if model_save_path:
        try:
            with open(model_save_path, 'wb') as f_out:
                pickle.dump(model, f_out)
            print(f"Character N-gram model saved to {model_save_path}")
        except Exception as e:
            print(f"Error saving character n-gram model: {e}")

    return model


def char_ngram_autocomplete(model, prompt, num_chars=20):
    """
    SImilar to ngram_autocomplete, predict a sequence of characters

    Args:
        model: A trained character-level n-gram model.
        prompt (str): Input text.
        num_chars (int): Number of characters to predict.

    Returns:
        str: The predicted continuation. Returns empty string if model is None/ cannot generate.
    """
    if model is None:
        return ""

    try:
        tokens = list(prompt)
        context = tokens[-(model.order - 1):] if model.order > 1 else []

        generated_chars = []
        for _ in range(num_chars):
            current_context = (generated_chars + tokens)[-(model.order - 1):] if model.order > 1 else []
            try:
                # text_seed expects list of char/ tokens 
                next_char = model.generate(text_seed=current_context)
                if not next_char or next_char == '</s>':  # Check for end token
                    break
                generated_chars.append(next_char)
            except Exception as e:
                # print(f"Char N-gram generation error for context {current_context}: {e}") # Optional debug
                break  

        return ''.join(generated_chars)
    except Exception as e:
        print(f"Error during char n-gram autocomplete: {e}")
        return ""


def load_ngram_model(model_path):
    """Loads a pickled n-gram model."""
    # error handlings
    try:
        with open(model_path, 'rb') as f_in:
            model = pickle.load(f_in)
        print(f"N-gram model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: N-gram model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading n-gram model: {e}")
        return None

