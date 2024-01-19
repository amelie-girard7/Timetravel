import os
from pathlib import Path
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.model_T5 import FlanT5FineTuner
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

    # Add a model checkpoint callback to save the model's best weights
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    
    # Add TensorBoard Logger
    logger = TensorBoardLogger("lightning_logs", name="flan-t5")

    trainer = pl.Trainer(
    max_epochs=CONFIG["max_epochs"],
    precision=16,
    accumulate_grad_batches=4,
    default_root_dir=model_save_path,
    callbacks=[checkpoint_callback],
    logger=logger,
    accelerator="gpu" if torch.cuda.is_available() else None,  # Use GPU if available
    devices=1 if torch.cuda.is_available() else None  # Number of GPUs
    )

    trainer.fit(model, train_dataloader, valid_dataloader)

    # Test the model after training
    trainer.test(model, test_dataloader)

    # Save the model manually (if needed, ModelCheckpoint saves the best automatically)
    # model.model.save_pretrained(model_save_path)
    # model.tokenizer.save_pretrained(model_save_path)

if __name__ == '__main__':
    main()
