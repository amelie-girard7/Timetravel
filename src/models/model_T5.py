# src/models/model_T5.py
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pytorch_lightning as pl
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import logging

from src.utils.config import CONFIG  # Import the CONFIG

logger = logging.getLogger(__name__)

class FlanT5FineTuner(pl.LightningModule):
    """
    This class defines a PyTorch Lightning model for fine-tuning a Flan-T5 model.
    It handles the forward pass, training, validation, and testing steps.
    """

    def __init__(self, model_name):
        """
        Initializes the model components, tokenizer, and metrics scorer.
        
        Args:
            model_name (str): The name of the T5 model to be used.
        """
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Initialize RougeScorer with all types of ROUGE metrics
        rouge_types = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        self.rouge_scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        self.rouge_types = rouge_types

        # Initialize a list to store outputs for each validation step
        self.current_val_step_outputs = []

    def forward(self, premise, initial, original_ending, counterfactual, labels=None, attention_mask=None):
        """
        Forward pass for the model.
        """
        # Concatenate the individual components to form input_ids
        input_ids = torch.cat([premise, initial, original_ending, counterfactual], dim=1)

        # Call the model's forward method
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        """
        Defines the training logic for a single batch, where a forward pass is performed and the loss is calculated.

        Args:
            batch (dict): The batch of data provided by the DataLoader.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        outputs = self.forward(
            premise=batch['premise'],
            initial=batch['initial'],
            original_ending=batch['original_ending'],
            counterfactual=batch['counterfactual'],
            labels=batch['edited_ending'],
            attention_mask=batch['attention_mask']
        )
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step for a single batch.
        It calculates the loss, generates predictions, and prepares data for metric calculation.
        
        Args:
            batch (dict): The batch of data provided by the DataLoader.
            batch_idx (int): The index of the current batch.
        """
        # Forward pass to compute loss and model outputs
        outputs = self.forward(
            premise=batch['premise'],
            initial=batch['initial'],
            original_ending=batch['original_ending'],
            counterfactual=batch['counterfactual'],
            labels=batch['edited_ending'],
            attention_mask=batch['attention_mask']
        )
        val_loss = outputs.loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Generate text predictions from the model using the individual components
        generated_texts = self.generate_text(
            premise=batch['premise'],
            initial=batch['initial'],
            original_ending=batch['original_ending'],
            counterfactual=batch['counterfactual'],
            attention_mask=batch['attention_mask']
        )

        # Decode the labels (ground truth edited ending) from the batch for comparison with the model's generated text.
        edited_endings = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['edited_ending']]

        # Store output information for metric calculation at the end of the epoch
        output = {
            'generated': generated_texts,
            'edited_endings': edited_endings,
            # Add other story components for later use in metric calculations
            'premises': [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['premise']],
            'counterfactuals': [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['counterfactual']],
            'original_endings': [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['original_ending']],
            'initials': [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['initial']],
        }
        self.current_val_step_outputs.append(output)


    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to calculate and log metrics.
        """
        # Initialize variables to store aggregated metrics
        aggregated_bleu_scores = 0
        aggregated_rouge_scores = {rouge_type: {"precision": 0, "recall": 0, "fmeasure": 0} for rouge_type in self.rouge_types}
        num_samples = len(self.current_val_step_outputs)  # Number of samples to average metrics over

        # Loop through each output from the validation steps
        for output in self.current_val_step_outputs:
            generated_texts = output['generated']
            edited_endings = output['edited_endings']
            
            # Calculate metrics for each generated-reference pair in the batch
            for gen, ref in zip(generated_texts, edited_endings):
                # Compute BLEU & ROUGE scores for each story component compared to the edited ending
                bleu_score = sentence_bleu([ref.split()], gen.split())
                aggregated_bleu_scores += bleu_score

                rouge_scores = self.rouge_scorer.score(ref, gen)
                for rouge_type, scores in rouge_scores.items():
                    aggregated_rouge_scores[rouge_type]["precision"] += scores.precision
                    aggregated_rouge_scores[rouge_type]["recall"] += scores.recall
                    aggregated_rouge_scores[rouge_type]["fmeasure"] += scores.fmeasure

        # Compute and log average BLEU & ROUGE scores across all validation data
        avg_bleu = aggregated_bleu_scores / num_samples
        self.log('avg_bleu', avg_bleu, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        for rouge_type in self.rouge_types:
            avg_precision = aggregated_rouge_scores[rouge_type]["precision"] / num_samples
            avg_recall = aggregated_rouge_scores[rouge_type]["recall"] / num_samples
            avg_fmeasure = aggregated_rouge_scores[rouge_type]["fmeasure"] / num_samples
            self.log(f'{rouge_type}_avg_precision', avg_precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{rouge_type}_avg_recall', avg_recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{rouge_type}_avg_fmeasure', avg_fmeasure, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Clear the list of outputs for the next epoch
        self.current_val_step_outputs = []


    def test_step(self, batch, batch_idx):
        """
        Called during the testing loop to perform a forward pass with a batch from the test set, calculate the loss, and optionally generate text.

        Args:
            batch (dict): The batch of data provided by the DataLoader.
            batch_idx (int): The index of the current batch.

        Returns:
            dict: Output dictionary containing generated texts and metrics.
        """
        # Perform forward pass and compute loss
        outputs = self.forward(
            premise=batch['premise'],
            initial=batch['initial'],
            original_ending=batch['original_ending'],
            counterfactual=batch['counterfactual'],
            labels=batch['edited_ending'],
            attention_mask=batch['attention_mask']
        )
        test_loss = outputs.loss
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Generate text using the model for the test batch.
        generated_texts = self.generate_text(
            premise=batch['premise'],
            initial=batch['initial'],
            original_ending=batch['original_ending'],
            counterfactual=batch['counterfactual'],
            attention_mask=batch.get('attention_mask')
        )
        
        # Decode the actual labels from the batch to get the ground truth text.
        decoded_targets = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['edited_ending']]
        
        print(f"\nTest Step {batch_idx} - Batch Keys: {batch.keys()}")  # Print the keys in the batch for debugging
        print(f"Generated Texts (first 2): {generated_texts[:2]}")  # Print the first 2 generated texts for inspection
        print(f"Decoded Targets (first 2): {decoded_targets[:2]}")  # Print the first 2 decoded targets for inspection
        
        # Calculate custom metrics using the calculate_metrics method.
        # TODO: calculate_metrics now logs the metrics internally. 
        self.calculate_metrics(generated_texts, decoded_targets)

        # Return the test loss
        return {'test_loss': test_loss}

    def calculate_metrics(self, generated_texts, edited_endings):
        """
        Calculates and logs metrics for the generated texts against the edited endings.
        """
        # Initialize variables to store aggregated metrics
        aggregated_bleu_score = 0
        aggregated_rouge_scores = {rouge_type: {"precision": 0, "recall": 0, "fmeasure": 0} for rouge_type in self.rouge_types}

        # Check if generated_texts and edited_endings are lists and have the same length
        if not isinstance(generated_texts, list) or not isinstance(edited_endings, list):
            self.log("error", "Both generated_texts and edited_endings must be lists.")
            return
        if len(generated_texts) != len(edited_endings):
            self.log("error", "generated_texts and edited_endings must have the same number of elements.")
            return

        # Calculate metrics for each generated-target pair
        for gen_text, target_text in zip(generated_texts, edited_endings):
            # Compute BLEU score
            try:
                bleu_score = sentence_bleu([target_text.split()], gen_text.split())
                aggregated_bleu_score += bleu_score
            except Exception as e:
                self.log("error", f"Error calculating BLEU score: {e}")

            # Compute ROUGE scores
            try:
                rouge_scores = self.rouge_scorer.score(target_text, gen_text)
                for rouge_type, scores in rouge_scores.items():
                    aggregated_rouge_scores[rouge_type]["precision"] += scores.precision
                    aggregated_rouge_scores[rouge_type]["recall"] += scores.recall
                    aggregated_rouge_scores[rouge_type]["fmeasure"] += scores.fmeasure
            except Exception as e:
                self.log("error", f"Error calculating ROUGE scores: {e}")

        # Compute average scores and log them
        num_samples = len(generated_texts)
        avg_bleu = aggregated_bleu_score / num_samples if num_samples > 0 else 0
        self.log('avg_bleu', avg_bleu, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        for rouge_type in self.rouge_types:
            avg_precision = aggregated_rouge_scores[rouge_type]["precision"] / num_samples if num_samples > 0 else 0
            avg_recall = aggregated_rouge_scores[rouge_type]["recall"] / num_samples if num_samples > 0 else 0
            avg_fmeasure = aggregated_rouge_scores[rouge_type]["fmeasure"] / num_samples if num_samples > 0 else 0
            self.log(f'{rouge_type}_avg_precision', avg_precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{rouge_type}_avg_recall', avg_recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{rouge_type}_avg_fmeasure', avg_fmeasure, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
        The optimizer is responsible for updating the model's weights to minimize the loss during training.
            
        Returns:
            The optimizer to be used for training the model.
        """
        #lr = CONFIG.get("learning_rate", 2e-5)  # Fetch the learning rate from CONFIG with a default
        #return torch.optim.AdamW(self.model.parameters(), lr=lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return optimizer

    def generate_text(self, premise, initial, original_ending, counterfactual, attention_mask=None, max_length=512):
        """
        Generates text sequences from the provided input components using the model.
        """
        # Concatenate the input components
        input_ids = torch.cat([premise, initial, original_ending, counterfactual], dim=1)

        # Generate a tensor of token IDs based on the input_ids and attention_mask
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
        
        # Decode the generated token IDs back into human-readable text
        generated_texts = [self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for generated_id in generated_ids]
        
        return generated_texts
