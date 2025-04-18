# src/model.py
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TextDataset,
    DataCollatorForLanguageModeling
)
import torch 
import os # used to check path exsistance
import warnings # to control warning during debug
import re


"""
This file contains work related to Transfer model. This file defines to load a transfor model
and its tokenizer from library or locally and generate autocomplete suggestions.
It also fune-tune the pre-trained model.

Functions:
1. setup_transformer_model(model_identifier="distilgpt2", cache_dir=None)
2. transformer_autocomplete(tokenizer, model, prompt, max_new_tokens=20, num_return_sequences=1, device=None)
3. create_text_dataset(file_path, tokenizer, block_size=64)
4. fine_tune_transformer(base_model_name="distilgpt2",train_file_path=None, eval_file_path=None, output_dir="./fine_tuned_model", epochs=1, batch_size=4, cache_dir=None)

Reference:
- CSPB 4830 Special Topic, Natural Language Processing, lecture and homework assignments
- https://web.stanford.edu/~jurafsky/slp3/
- https://huggingface.co/docs/transformers/en/training
- https://github.com/huggingface/transformers/tree/main/notebooks
"""


# Suppress specific warnings as there were originally many
warnings.filterwarnings("ignore", message=".*Using pad_token_id.*")


def setup_transformer_model(model_identifier="distilgpt2", cache_dir=None):
    """
    This function setup transformer model pipeline for autocomplete.
    Loads either a pre-trained model from Hugging Face or a fine-tuned model from a local path.

    Args:
        model_identifier (str): Pre-trained model name (e.g., 'distilgpt2') or path to a fine-tuned model directory.
        cache_dir (str, optional): Directory to cache downloaded models. Defaults to None (Hugging Face default).

    Returns:
        tuple: (tokenizer, model, device) or (None, None, None) if loading fails.
    """
    try: # error handling
        print(f"Loading transformer model: {model_identifier}")
        
        device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU with pipeline
        print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

        tokenizer = AutoTokenizer.from_pretrained(model_identifier, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_identifier, cache_dir=cache_dir)

        # Add pad token if missing (reference from hugging face)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
            print("Added pad_token identical to eos_token.")

        print("Tokenizer and model loaded successfully.")
        return tokenizer, model, device

    except OSError as e:
        print(
            f"Error loading model/tokenizer '{model_identifier}'. Is it a valid Hugging Face model name or local path? Error: {e}")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during model setup: {e}")
        return None, None, None


def transformer_autocomplete(tokenizer, model, prompt, max_new_tokens=20, num_return_sequences=1, device=None):
    """
    Generate autocomplete suggestions using a loaded transformer model.

    Args:
        tokenizer: The loaded tokenizer.
        model: The loaded model.
        prompt (str): The text input (should be cleaned but not necessarily lowercased unless model expects it).
        max_new_tokens (int): Max number of *new* tokens (approx words/subwords) to generate.
        num_return_sequences (int): Number of suggestions to generate (usually 1 for evaluation).
        device (torch.device, optional): The device to run inference on.

    Returns:
        str: The top generated continuation string. Returns empty on failure.
    """
    if not tokenizer or not model:
        print("Error: Tokenizer or model not provided for autocomplete.")
        return ""

    try:
        current_device = model.device
        if device is not None and current_device != device:
            model.to(device)
            print(f"Moved model to device: {device}")

        # Encode the prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=tokenizer.model_max_length - max_new_tokens)

        # Move inputs to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        # Use 'attention_mask' for potentially padded inputs if tokenizer adds padding
        output_sequences = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),  # Use attention mask if available
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,  # Prevent simple repetitions
            early_stopping=True,  # Stop when EOS token is generated
            pad_token_id=tokenizer.eos_token_id  # Set pad_token_id for generation
        )

        # Decode the generated tokens, skipping the prompt tokens and special tokens
        generated_texts = tokenizer.batch_decode(output_sequences[:, inputs['input_ids'].shape[-1]:],
                                                 skip_special_tokens=True)

        # Return the first generated sequence's continuation
        if generated_texts:
            # Basic cleanup of the generated text
            first_suggestion = generated_texts[0].strip()
            first_suggestion = re.sub(r"^[.,;!?'\"]+", "", first_suggestion).strip()
            return first_suggestion
        else:
            return ""

    except Exception as e:
        print(f"Error during transformer autocomplete for prompt '{prompt[:50]}...': {e}")
        return ""


# helper function for fine tuning
def create_text_dataset(file_path, tokenizer, block_size=64): #changed block sizee from 128 to reduce RAM/time 
    """Helper to create dataset for Trainer"""
    try:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=block_size  # Max sequence length for training chunks
        )
    except Exception as e:
        print(f"Error creating TextDataset from {file_path}: {e}")
        return None


def fine_tune_transformer(
        base_model_name="distilgpt2",
        train_file_path=None,  # Path to train text file (e.g., train_df['clean_text'].to_csv('train.txt'))
        eval_file_path=None,  # Path to validation text file
        output_dir="./fine_tuned_model",
        epochs=1,
        batch_size=4,
        cache_dir=None):
  
    print("--- Starting Transformer Fine-Tuning ---")

    # check input
    if not train_file_path or not eval_file_path:
        print("Error: Training and validation file paths are required for fine-tuning.")
        return None

    if not os.path.exists(train_file_path) or not os.path.exists(eval_file_path):
        print(f"Error: Ensure train ({train_file_path}) and eval ({eval_file_path}) files exist.")
        return None

    # load model
    tokenizer, model, _ = setup_transformer_model(base_model_name, cache_dir=cache_dir)
    if not tokenizer or not model:
        print("Failed to load base model for fine-tuning.")
        return None

    print("Preparing datasets...")
    train_dataset = create_text_dataset(train_file_path, tokenizer)
    eval_dataset = create_text_dataset(eval_file_path, tokenizer)

    # Check if dataset creatied
    if not train_dataset or not eval_dataset:
        print("Failed to create datasets. Exiting fine-tuning.")
        return None
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)  # MLM=False for Causal LM

    print("Setting up Training Arguments...")
    # parameters are tuned/ revised to fine tune successfully.
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir=os.path.join(output_dir, 'logs'), # logging_dir='./logs',  # Directory for logs
        logging_steps=100,  # Log metrics every 100 steps
        learning_rate=5e-5,  # Common starting learning rate
        weight_decay=0.01,  # Weight decay for regularization
        warmup_steps=100,  # Number of warmup steps for learning rate scheduler
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available (requires accelerate)
        report_to="tensorboard", 
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting fine-tuning training...")
    try:
        trainer.train()
        print("Fine-tuning completed.")

        print(f"Saving fine-tuned model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)  # Save tokenizer alongside model
        print("Model and tokenizer saved.")
        return output_dir  # Return path to the fine-tuned model

    except Exception as e:
        print(f"An error occurred during fine-tuning: {e}")
        return None


if __name__ == "__main__":
    print("\n--- Testing Pre-trained Model ---")

    model_id = "distilgpt2"  # gpt2 didn't work with given CPU limit

    tokenizer, model, device = setup_transformer_model(model_id)

    if tokenizer and model:
        prompt_example = "dear team, please find attached the latest"
        # Use the direct generation function
        suggestions = transformer_autocomplete(tokenizer, model, prompt_example, max_new_tokens=15,
                                               num_return_sequences=1, device=model.device)

        print(f"\nPrompt: '{prompt_example}'")
        print(f"Transformer Suggestion: '{suggestions}'")

    else:
        print("Failed to load the model for testing.")

