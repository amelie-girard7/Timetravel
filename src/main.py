import os
from pathlib import Path
import sys
import pytorch_lightning as pl

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.model import FlanT5FineTuner
from src.data_loader import create_dataloaders
from src.utils.utils import preprocess_data, collate_fn

CONFIG = {
    "root_dir": Path(__file__).resolve().parent.parent,
    "model_name": "google/flan-t5-base",
    "batch_size": 4,
    "num_workers": 3,
    "max_epochs": 3,
}

# Define the model globally
model = FlanT5FineTuner(CONFIG["model_name"])

def custom_collate_fn(batch):
    # Use the globally defined model's tokenizer
    return collate_fn(batch, model.tokenizer)

def main():
    data_path = CONFIG["root_dir"] / 'data' / 'raw'
    model_save_path = CONFIG["root_dir"] / 'models'
    model_save_path.mkdir(exist_ok=True)

    file_names = ['train_supervised_small1.json', 'dev_data1.json', 'test_data1.json']

    dataloaders = create_dataloaders(
        data_path,
        file_names,
        CONFIG["batch_size"],
        custom_collate_fn,  # Use the custom collate function
        preprocess_data,
        CONFIG["num_workers"]
    )

    train_dataloader = dataloaders['train_supervised_small1.json']
    valid_dataloader = dataloaders['dev_data1.json']
    test_dataloader = dataloaders['test_data1.json']

    trainer = pl.Trainer(
        max_epochs=CONFIG["max_epochs"],
        precision=16,
        accumulate_grad_batches=4,
        default_root_dir=model_save_path
    )

    trainer.fit(model, train_dataloader, valid_dataloader)

    model.model.save_pretrained(model_save_path)
    model.tokenizer.save_pretrained(model_save_path)

if __name__ == '__main__':
    main()
