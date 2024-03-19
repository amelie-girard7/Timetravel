# src/utils/utils.py

#from fileinput import filename
import json
import logging
import torch
import torch.nn.utils.rnn


# Global constant for padding token ID, set to 0 by default for T5.
PAD_TOKEN_ID = 0

# # Configure a module-level logger using the standard Python logging library.
logger = logging.getLogger(__name__)

def count_json_lines(file_path):
    """
    Counts the number of lines in a JSON file. This can be useful for estimating
    the size of the dataset and for iteration purposes.
    """
    print(f"Counting lines in file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return sum(1 for _ in file)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    

def load_first_line_from_json(file_path):
    """
    Loads and parses the first line from a JSON file. Useful for quickly inspecting
    the structure of the data without needing to load the entire file into memory.
    """
    print(f"Loading first line from JSON file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.loads(next(file))
    except Exception as e:
        logger.error(f"Error reading from {file_path}: {e}")
        raise IOError(f"Error reading from {file_path}: {e}")

def preprocess_data(row, tokenizer, max_length=512):
    """
    Preprocesses a single row of data to format it for input to the T5 model. This includes
    constructing the input sequence from the various components of the data and tokenizing it
    along with the output sequence.
    """
    print("\nPreprocessing data row...")
    
    try:
        # Define the separator token specific to the T5 model.
        separator_token = "</s>"
        
        # Construct the input sequence with all components separated by the model-specific separator token.
        input_sequence = (
            f"{row['premise']}"
            f"{row['initial']}"
            f"{row['original_ending']} {separator_token} "
            f"{row['premise']} {row['counterfactual']}"
        )
        print(f"Constructed input sequence: {input_sequence[:128]}...")
        
        # Tokenize the input sequence with truncation to max_length and no padding here.
        tokenized_inputs = tokenizer.encode_plus(
            input_sequence, truncation=True, return_tensors="pt", max_length=max_length
        )
        print("Tokenized input sequence.")
        
        # Join the list of edited endings into a single string
        edited_ending_joined = ' '.join(row['edited_ending'])
        # edited_ending_joined = ' '.join(row['edited_ending'] if isinstance(row['edited_ending'], list) else [row['edited_ending']])
        print(f"Constructed edited ending sequence: {edited_ending_joined[:50]}...")
        
        # Tokenize the output sequence (edited ending) with truncation to max_length.
        tokenized_ending = tokenizer.encode_plus(
            edited_ending_joined, truncation=True, return_tensors="pt", max_length=max_length
        )
        print("Tokenized edited ending sequence.")
        
        # Return the tokenized inputs, labels, and original data fields for evaluation.
        return {
            'input_ids': tokenized_inputs['input_ids'].squeeze(0),
            'attention_mask': tokenized_inputs['attention_mask'].squeeze(0),
            'labels': tokenized_ending['input_ids'].squeeze(0),
            # Include non-tokenized data for metric calculations.
            'premise': row['premise'],
            'initial': row['initial'],
            'original_ending': row['original_ending'],
            'counterfactual': row['counterfactual'],
            'edited_ending': edited_ending_joined
        }
    except Exception as e:
        logger.error(f"Error in preprocess_data: {e}")
        return None


def collate_fn(batch):
    """
    Custom collate function to pad and combine a batch of preprocessed data into
    tensor formats suitable for model input. This function is specifically tailored
    for batches of data that have been preprocessed with `preprocess_data`.
    """
    print(f"Amelie: Collating batch of size {len(batch)}")
    
    # Unpack the batch into separate lists for each field.
    input_ids, attention_mask, labels, premise, initial, original_ending, counterfactual, edited_ending = list(zip(*batch))
    
    # Pad the sequences for 'input_ids', 'attention_mask', and 'labels' to the longest in the batch.
    # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
    # TODO: Try to pass the PAD_TOKEN_ID as a parameter to the collate_fn
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
    print("Padded input_ids.")
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    print("Padded attention_mask.")
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PAD_TOKEN_ID)
    print("Padded labels.")

    print("Batch collated successfully.")
    
    # Return the padded tensors along with the additional fields for evaluation.
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        # Additional fields for evaluation
        'premise': premise,
        'initial': initial,
        'original_ending': original_ending,
        'counterfactual': counterfactual,
        'edited_ending': edited_ending,
    }
