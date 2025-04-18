# Text Generation for Autocomplete Using Transformer Models

## 1. Project Overview

This project explores the use of Transformer models for generating email autocomplete suggestions. 
It compares the performance of a pre-trained `distilgpt2` model with the fine-tuned model on a subset of the Enron email dataset. Simple N-gram models (word-level and character-level trigrams) are implemented as baselines. The evaluation focuses on next token prediction accuracy and Keystroke Savings Rate (KSR). An interactive demo using Gradio showcases the fine-tuned model.

## 2. Project Structure
```
project_root/
├── data/
│   └── emails.csv              # Raw data source (Not included - please see Setup section)
├── notebooks/
│   ├── email_text_autocomplete.ipynb  # Main execution notebook (eval, fine-tuning)
│   └── Project_Demo_Gradio.ipynb      # Interactive Gradio demo notebook
├── src/
│   ├── init.py                 # Makes 'src' a Python package
│   ├── data_preprocessing.py   # Data loading, cleaning, splitting functions
│   ├── evaluation.py           # Accuracy & KSR metric calculation functions
│   ├── baseline_ngram.py       # N-gram baseline model implementation
│   └── model.py                # Transformer model implementation (setup, predict, fine-tune)
│   └── test_evaluation.py      # Main script to run evaluation pipeline
└── README.md

--- Generated Outputs (Created by running notebooks/scripts) ---
fine_tuned_distilgpt2_subset_test/ # Saved fine-tuned model
evaluation_results_detailed.csv # Detailed evaluation results
char_ngram_model.pkl # Saved N-gram model for char to char
word_ngram_model.pkl # Saved N-gram model for word to word
train.txt # Generated: Full training text
val.txt # Generated: Full validation text
train_subset.txt # Generated: Subset training text
val_subset.txt # Generated: Subset validation text               
```

## 3. Setup

### Prerequisites
* Python 3.9 or higher
* Access to Google Colab or a local environment with high CPU/GPU, RAM.

### Data Acquisition
1.  Download the Enron Email Dataset from Kaggle: [https://www.kaggle.com/datasets/wcukierski/enron-email-dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)
2.  Place the `emails.csv` file into a `data/` directory within the project root (e.g., `project_root/data/emails.csv`). *Note: The file size is ~1.5GB*

### Installation & Download
1. **Clone the repository:**
    ```bash
    git clone https://github.com/Mayumi-GT/email-autocomplete-transformer
    cd email-autocomplete-transformer
    ```
2. **Open the `Email_Text_Autocomplete.ipynb` notebook.**
    *Core dependencies include:* `pandas`, `scikit-learn`, `nltk`, `transformers`, `torch`, `tqdm`, `huggingface_hub`, `accelerate`, `ipywidgets`, `gradio`.
    Google Colab comes with many of them pre-installed

## 4. How to Run

The primary way to run this project is through the Google Colab notebooks provided in the `notebooks/` directory.

### A. Running Evaluation & Fine-Tuning (`email_text_autocomplete.ipynb`)

1.  **Open in Colab:** Open the `email_text_autocomplete.ipynb` notebook from the `notebooks` folder.
2.  **Set Runtime Type:** For fine-tuning, ensure a **GPU runtime** (e.g., T4) is selected (Runtime -> Change runtime type). Evaluation can run on CPU but will be slower for the Transformer.
3.  **Adjust Paths:** Verify the `DATA_FILE_PATH` inside the notebook cells or within `src/test_evaluation.py` correctly points to your `emails.csv` location on Drive after mounting (usually `/content/drive/MyDrive/project/emails.csv`). Also check paths for saving/loading models if you modify them.
4.  **Run Cells Sequentially:** Execute the notebook cells one by one, following the Markdown instructions within the notebook:
    * Mount Drive & Authorize.
    * Install libraries & Download NLTK data.
    * Change directory (`%cd`) to your project root on Drive.
    * Set environment variables.
    * **Run Evaluation:** Execute the `!python -m src.test_evaluation` cell. *Before running*, edit `src/test_evaluation.py` to set `TRANSFORMER_MODEL_ID` to either `"distilgpt2"` (pre-trained) or the path to your fine-tuned model (e.g., `/content/drive/MyDrive/project/fine_tuned_distilgpt2_subset_test`).
    * **Prepare Data:** Run the cells to create `train.txt`, `val.txt`, and the subset files (`train_subset.txt`, `val_subset.txt`). This only needs to be done once.
    * **Run Fine-Tuning:** Execute the cell that calls `fine_tune_transformer`. This uses the **subset data** by default in the provided notebook code and saves the model to `/content/drive/MyDrive/project/fine_tuned_distilgpt2_subset_test`. This step takes 1-1.5 hr with T4 GPU.

### B. Running the Demo (`Project_Demo_Gradio.ipynb`)

1.  **Open in Colab:** Open the `Project_Demo_Gradio.ipynb` notebook.
2.  **Set Runtime Type:** A CPU runtime is usually sufficient and stable for the Gradio demo.
3.  **Adjust Model Path:** Verify the `model_path` variable in the notebook points to your saved fine-tuned model directory (e.g., `/content/drive/MyDrive/project/fine_tuned_distilgpt2_subset_test`).
4.  **Run Cells Sequentially:** Execute the notebook cells:
    * Mount Drive & Setup.
    * Load the Fine-Tuned Model.
    * Define the Gradio function.
    * Launch the Gradio Interface.
5.  **Interact:** Use the interface provided in the cell output or the public `.gradio.live` link generated.
