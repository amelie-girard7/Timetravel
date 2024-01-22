import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from functools import partial
from src.utils.utils import preprocess_data, collate_fn

class CustomJSONDataset(Dataset):
    """
    A PyTorch Dataset class that handles loading and preprocessing data from JSON files.
    
    Attributes:
        processed_data (DataFrame): Contains the preprocessed data ready for model input.
    """

    def __init__(self, file_path, preprocess_fn):
        """
        Initializes the dataset object by reading and preprocessing the JSON data.
        
        Parameters:
            file_path (str or Path): The file path to the JSON file containing the raw data.
            preprocess_fn (function): The preprocessing function to apply to the data.
        
        Raises:
            ValueError: If there is an error reading or parsing the JSON file.
            FileNotFoundError: If the JSON file cannot be found at the specified path.
        """
        try:
            # Read the JSON file into a pandas DataFrame
            data = pd.read_json(file_path, lines=True)
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing {file_path}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        # Apply the preprocessing function to each row of the DataFrame
        self.processed_data = data.apply(preprocess_fn, axis=1, result_type='expand')

    def __len__(self):
        """Returns the total number of items in the dataset."""
        return len(self.processed_data)

    def __getitem__(self, idx):
        """
        Retrieves an item by its index from the dataset.
        
        Parameters:
            idx (int): Index of the item.
        
        Returns:
            Series: The processed data at the given index.
        """
        return self.processed_data.iloc[idx]

# Define a new collate function that takes tokenizer as a parameter
def custom_collate_fn(batch, tokenizer):
    return collate_fn(batch, tokenizer)

def create_dataloaders(data_path, file_names, batch_size, tokenizer, preprocess_fn, num_workers=0):
    """
    Creates a dictionary of DataLoader objects for each specified JSON file.
    
    Parameters:
        data_path (Path): Path to the directory where data files are stored.
        file_names (list of str): List of JSON file names to create dataloaders for.
        batch_size (int): Number of samples to be loaded per batch.
        tokenizer (PreTrainedTokenizer): The tokenizer used for encoding the text data.
        preprocess_fn (function): Preprocessing function to be applied to each data item.
        num_workers (int): Number of worker threads to use for data loading (default is 0, which means the data will be loaded in the main process).
    
    Returns:
        dict: A dictionary mapping file names to their respective DataLoader.
    
    Raises:
        FileNotFoundError: If a specified file does not exist in the data directory.
    """
    dataloaders = {}
    for file_name in file_names:
        # Construct the full path to the JSON file
        file_path = Path(data_path) / file_name
        # Raise an error if the file doesn't exist
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} does not exist.")

        # Create a custom dataset using the JSON file and the preprocessing function
        dataset = CustomJSONDataset(file_path, preprocess_fn)
        # Create a partial function for collate_fn with tokenizer
        collate_fn_with_tokenizer = partial(custom_collate_fn, tokenizer=tokenizer)
        # Create a DataLoader for batching and loading the dataset
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=collate_fn_with_tokenizer,  # Use the partial function
            num_workers=num_workers
        )
        # Store the DataLoader in the dictionary using the file name as the key
        dataloaders[file_name] = dataloader

    return dataloaders