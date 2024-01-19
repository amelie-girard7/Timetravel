import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from pathlib import Path

class CustomJSONDataset(Dataset):
    """
    Custom Dataset class for handling JSON files.
    """

    def __init__(self, file_path, preprocess_fn):
        """
        Initialize the dataset with a file path and a preprocessing function.
        """
        try:
            data = pd.read_json(file_path, lines=True)
        except Exception as e:
            raise ValueError(f"Error reading {file_path}: {e}")
        
        self.processed_data = data.apply(preprocess_fn, axis=1, result_type='expand')

    def __len__(self):
        """
        Return the number of items in the dataset.
        """
        return len(self.processed_data)

    def __getitem__(self, idx):
        """
        Retrieve an item by its index.
        """
        return self.processed_data.iloc[idx]

def create_dataloaders(data_path, file_names, batch_size, collate_fn, preprocess_fn, num_workers=0):
    """
    Create dataloaders for each file in file_names.
    """
    dataloaders = {}
    for file_name in file_names:
        file_path = Path(data_path) / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} does not exist.")
        
        dataset = CustomJSONDataset(file_path, preprocess_fn)
        hf_dataset = HFDataset.from_pandas(dataset.processed_data)
        dataloader = DataLoader(
            hf_dataset, 
            batch_size=batch_size, 
            collate_fn=collate_fn,
            num_workers=num_workers
        )
        dataloaders[file_name] = dataloader
    return dataloaders