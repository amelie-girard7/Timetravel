import json
import pandas as pd

def count_json_lines(file_path):
    """
    Count the number of lines in a JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return sum(1 for _ in file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

def load_first_line_from_json(file_path):
    """
    Load the first line from a JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.loads(next(file))
    except Exception as e:
        raise IOError(f"Error reading from {file_path}: {e}")

def preprocess_data(row):
    """
    Preprocess a row of the dataset for Flan-T5 model.
    Combines 'premise', 'initial', and 'counterfactual' into a single string.
    Uses the first 'edited_ending' as the target output, if available.
    """
    input_text = f"Premise: {row['premise']} Initial: {row['initial']} Counterfactual: {row['counterfactual']}"
    output_text = row.get('edited_ending', [""])[0] if isinstance(row.get('edited_ending', [""]), list) else row.get('edited_ending', "")
    return pd.Series({"input_text": input_text, "output_text": output_text})

def collate_fn(batch, tokenizer):
    """
    Tokenize and collate a batch of data for the T5 model.
    """
    input_texts = [item['input_text'] for item in batch]
    output_texts = [item['output_text'] for item in batch]
    encoding = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(output_texts, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    encoding["labels"] = labels
    return encoding