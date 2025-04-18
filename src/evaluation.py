# src/evaluation.py

"""
This file contains functions to calculate performance matrics for autocomplete models.
It takes the text the model should have suggested, ground truth, and the text 
the model actually suggested, and computes numbers representing accuracy and keystroke saving rate.

Functions:
1. get_first_word(text)
2. get_first_char(text)
3. compute_next_word_accuracy(ground_truth_continuation, suggestion)
4. compute_next_char_accuracy(ground_truth_continuation, suggestion)
5. calculate_keystroke_savings_rate(ground_truth_continuation, suggestion, min_match_len=3)

Reference:
- https://www.nltk.org/bookLinks
- https://web.stanford.edu/~jurafsky/slp3/
- Trnka, K., & McCoy, K. F. (2008). Evaluating word prediction: Framing keystroke savings. 
- Koester, H. H., & Levine, S. P. (1994). Modeling the speed of text entry with a word prediction interface
"""

def get_first_word(text):
    """Extracts the first word from a string."""
    words = text.split()
    return words[0] if words else ""


def get_first_char(text):
    """Extracts the first character from a string."""
    return text[0] if text else ""


def compute_next_word_accuracy(ground_truth_continuation, suggestion):
    """
    Calculates whether the model's very first suggested word matches the actual 
    first word that should have come next, the ground truth continuation. This 
    gives a simple "yes/no" (1/0) score for predicting the immediate next word.

    Args:
        ground_truth_continuation (str): The actual text that should follow the prompt.
        suggestion (str): The text suggested by the model.

    Returns:
        int: 1 if the first word matches (case-insensitive), 0 otherwise.
    """
    true_first_word = get_first_word(ground_truth_continuation.strip().lower())
    predicted_first_word = get_first_word(suggestion.strip().lower())

    # edge case if ground truth is emoty
    if not true_first_word:  
        return 0

    return 1 if true_first_word == predicted_first_word else 0


def compute_next_char_accuracy(ground_truth_continuation, suggestion):
    """
    Similar to the word accuracy above, computes accuracy based on whether 
    the first predicted character matches the first character of the ground truth continuation.

    Args:
        ground_truth_continuation (str): The actual text that should follow the prompt.
        suggestion (str): The text suggested by the model.

    Returns:
        int: 1 if the first character matches (case-insensitive), 0 otherwise.
    """
    true_first_char = get_first_char(ground_truth_continuation.strip().lower())
    predicted_first_char = get_first_char(suggestion.strip().lower())

    if not true_first_char:  # edge case
        return 0

    return 1 if true_first_char == predicted_first_char else 0


def calculate_keystroke_savings_rate(ground_truth_continuation, suggestion, min_match_len=3):
    """
    This calculates KSR, Keystroke Savings Rate.
    Assumes the user accepts the suggestion *only if* it correctly predicts
    at least `min_match_len` starting characters of the ground truth continuation.

    Args:
        ground_truth_continuation (str): The actual text the user would type.
        suggestion (str): The text suggested by the model.
        min_match_len (int): Minimum number of initial characters that must match
                             for the suggestion to be considered "accepted".

    Returns:
        float: The Keystroke Savings Rate (0.0 to 1.0). Returns 0.0 if ground truth is empty.
    """
    ground_truth = ground_truth_continuation.strip()
    suggestion = suggestion.strip()

    # edge case
    if not ground_truth:
        return 0.0  

    actual_keystrokes_needed = len(ground_truth)

    # Check if the suggestion provides a useful match at the beginning
    matches = False
    saved_keystrokes = 0

    # Normalize for comparison 
    norm_ground_truth = ground_truth.lower()
    norm_suggestion = suggestion.lower()

    # Check for a prefix match of at least min_match_len chars
    # There are two points to check: 1: is suggestion long enough? and 2.
    # does the ground truth actually start with the the first min_match_len characters?
    if len(norm_suggestion) >= min_match_len and \
            norm_ground_truth.startswith(norm_suggestion[:min_match_len]):

        # Find the longest common prefix between suggestion and ground truth
        common_prefix_len = 0
        for i in range(min(len(norm_ground_truth), len(norm_suggestion))):
            if norm_ground_truth[i] == norm_suggestion[i]:
                common_prefix_len += 1
            else:
                break # stops as soon as a mismatch occurs.

        # Only consider it a "match" if the common prefix meets the minimum length
        if common_prefix_len >= min_match_len:
            matches = True
            # How many keystrokes are saved if accepted?
            # User types 1 keystroke (e.g., Tab) to accept the common prefix.
            # Saved = (chars typed without suggestion) - (chars typed with suggestion)
            # Saved = common_prefix_len - 1
            # Calculate savings relative to typing the *entire* ground truth
            saved_keystrokes = common_prefix_len - 1  # Subtract 1 for the acceptance keystroke

    # edge case where saved_keystrokes isn't negative
    saved_keystrokes = max(0, saved_keystrokes)

    # Calculate KSR
    # KSR = (Keystrokes Saved) / (Total Keystrokes Needed Without Suggestion)
    ksr = saved_keystrokes / actual_keystrokes_needed if actual_keystrokes_needed > 0 else 0.0

    # Ensure KSR is between 0 and 1 (it could exceed 1 if suggestion is longer than GT but matches prefix)
    # In practice, you likely wouldn't save more than typing the GT itself.
    # However, the metric reflects the *potential* savings from the matched prefix. Let's cap it at 1.0?
    # Or perhaps the number of saved keystrokes should be capped at `actual_keystrokes_needed - 1`?
    # Let's cap savings at `actual_keystrokes_needed - 1`

    if saved_keystrokes >= actual_keystrokes_needed:
        saved_keystrokes = actual_keystrokes_needed - 1  # Max saving is typing everything except 1 char + accept key
        saved_keystrokes = max(0, saved_keystrokes)  # Ensure non-negative

    ksr = saved_keystrokes / actual_keystrokes_needed if actual_keystrokes_needed > 0 else 0.0

    return max(0.0, min(ksr, 1.0))  # Ensure KSR is within [0, 1]

# this is to test each function above
if __name__ == "__main__":
    # --- Test Evaluation Functions ---
    gt = "the quick brown fox jumps over the lazy dog"
    sugg_good_word = "the lazy cat sleeps"
    sugg_good_char = "that quick movement"
    sugg_bad = "a totally different sentence"
    sugg_exact_prefix = "the quick brown fox"
    sugg_long_match = "the quick brown fox jumps over the lazy dog and then some more"
    sugg_short = "th"

    print(f"Ground Truth: '{gt}'")
    print("-" * 20)

    # Accuracy Tests
    print("Next Word Accuracy:")
    print(
        f"Suggestion: '{sugg_good_word}' -> Match: {compute_next_word_accuracy(gt, sugg_good_word)}")  # Should be 1 (the == the)
    print(
        f"Suggestion: '{sugg_good_char}' -> Match: {compute_next_word_accuracy(gt, sugg_good_char)}")  # Should be 0 (the != that)
    print(f"Suggestion: '{sugg_bad}' -> Match: {compute_next_word_accuracy(gt, sugg_bad)}")  # Should be 0 (the != a)
    print(
        f"Suggestion: '{sugg_short}' -> Match: {compute_next_word_accuracy(gt, sugg_short)}")  # Should be 0 (the != th)

    print("\nNext Char Accuracy:")
    print(
        f"Suggestion: '{sugg_good_word}' -> Match: {compute_next_char_accuracy(gt, sugg_good_word)}")  # Should be 1 (t == t)
    print(
        f"Suggestion: '{sugg_good_char}' -> Match: {compute_next_char_accuracy(gt, sugg_good_char)}")  # Should be 1 (t == t)
    print(f"Suggestion: '{sugg_bad}' -> Match: {compute_next_char_accuracy(gt, sugg_bad)}")  # Should be 0 (t != a)
    print(f"Suggestion: '{sugg_short}' -> Match: {compute_next_char_accuracy(gt, sugg_short)}")  # Should be 1 (t == t)

    # KSR Tests (min_match_len = 3)
    print("\nKeystroke Savings Rate (min_match_len=3):")
    print(
        f"Suggestion: '{sugg_good_word}' -> KSR: {calculate_keystroke_savings_rate(gt, sugg_good_word, 3):.4f}")  # Matches "the", len=3. Saved=3-1=2. KSR=2/len(gt)
    print(
        f"Suggestion: '{sugg_good_char}' -> KSR: {calculate_keystroke_savings_rate(gt, sugg_good_char, 3):.4f}")  # Matches "th", len=2. Less than min_match_len=3. KSR=0.0
    print(
        f"Suggestion: '{sugg_bad}' -> KSR: {calculate_keystroke_savings_rate(gt, sugg_bad, 3):.4f}")  # No match. KSR=0.0
    print(
        f"Suggestion: '{sugg_exact_prefix}' -> KSR: {calculate_keystroke_savings_rate(gt, sugg_exact_prefix, 3):.4f}")  # Matches "the quick brown fox", len=19. Saved=19-1=18. KSR=18/len(gt)
    print(
        f"Suggestion: '{sugg_long_match}' -> KSR: {calculate_keystroke_savings_rate(gt, sugg_long_match, 3):.4f}")  # Matches all of gt, len=43. Saved=43-1=42. KSR=42/len(gt)
    print(
        f"Suggestion: '{sugg_short}' -> KSR: {calculate_keystroke_savings_rate(gt, sugg_short, 3):.4f}")  # Matches "th", len=2. Less than min_match_len=3. KSR=0.0
    print(
        f"Suggestion: 'the' -> KSR: {calculate_keystroke_savings_rate(gt, 'the', 3):.4f}")  # Matches "the", len=3. Saved=3-1=2. KSR=2/len(gt)
