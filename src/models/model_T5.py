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

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Called during training and validation steps to perform a forward pass through the model.

        Args:
            input_ids (torch.Tensor): Tensor of token ids to be fed to the model.
            attention_mask (torch.Tensor, optional): Tensor of attention mask to be fed to the model.
            labels (torch.Tensor, optional): Tensor of labels for calculating loss.

        Returns:
            transformers.modeling_outputs.Seq2SeqLMOutput: Output from the model which includes loss when labels are provided.
        """
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        """
        Defines the training logic for a single batch, where a forward pass is performed and the loss is calculated.

        Args:
            batch (dict): The batch of data provided by the DataLoader.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        outputs = self.forward(**batch)
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
            
        Returns:
            dict: A dictionary containing loss, predictions, and target texts for the current batch.
        """
        print(batch.keys())  # Add this line to check what keys are available in the batch
        
        # Forward pass to compute loss and model outputs
        outputs = self.forward(input_ids=batch['input_ids'], 
                            attention_mask=batch['attention_mask'], 
                            labels=batch['labels'])
        val_loss = outputs.loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Generate text predictions from the model using the input_ids
        generated_texts = self.generate_text(batch['input_ids'], batch.get('attention_mask'))
        
        # Decode the labels (ground truth edited ending) from the batch for comparison with the model's generated text.
        edited_endings = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['labels']]

        # Decode the individual components for comparison
        premises = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['premise']]
        counterfactuals = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['counterfactual']]
        original_endings = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['original_ending']]

        # Store output information for metric calculation at the end of the epoch
        output = {
            'generated': generated_texts,
            'edited_endings': edited_endings,
            'premises': premises,
            'counterfactuals': counterfactuals,
            'original_endings': original_endings
        }
        self.current_val_step_outputs.append(output)

    def on_validation_epoch_end(self):
        """
        Called at the end of a validation epoch.
        It aggregates the results from all batches and calculates average BLEU and ROUGE scores.
        """
        # Initialize variables to store aggregated metrics
        aggregated_bleu_scores = 0
        aggregated_rouge_scores = {
            rouge_type: {"precision": 0, "recall": 0, "fmeasure": 0} 
            for rouge_type in self.rouge_types
        }
        
        # Calculate the number of samples over which to average the metrics
        num_samples = len(self.current_val_step_outputs)
        
        # Loop through each output from the validation steps
        for output in self.current_val_step_outputs:
            generated_texts = output['generated']
            edited_endings = output['edited_endings']
            premises = output['premises']
            counterfactuals = output['counterfactuals']
            original_endings = output['original_endings']

            # Calculate metrics for each generated-target pair in the batch
            for gen, edited_ending, premise, counterfactual, original_ending in zip(generated_texts, edited_endings, premises, counterfactuals, original_endings):
                # Compute and log BLEU & ROUGE scores for each story component compared to the edited ending
                for comparison_type, ref_text in zip(['premise', 'counterfactual', 'original_ending'], [premise, counterfactual, original_ending]):
                    bleu_score = sentence_bleu([edited_ending.split()], ref_text.split())
                    aggregated_bleu_scores += bleu_score

                    rouge_scores = self.rouge_scorer.score(edited_ending, ref_text)
                    for rouge_type, scores in rouge_scores.items():
                        aggregated_rouge_scores[rouge_type]["precision"] += scores.precision
                        aggregated_rouge_scores[rouge_type]["recall"] += scores.recall
                        aggregated_rouge_scores[rouge_type]["fmeasure"] += scores.fmeasure

        # Compute average BLEU and ROUGE scores across all validation data
        avg_bleu = aggregated_bleu_scores / (num_samples * 3)  # Multiply by 3 because we have 3 comparisons per sample
        self.log('avg_bleu', avg_bleu, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        for rouge_type in self.rouge_types:
            avg_precision = aggregated_rouge_scores[rouge_type]["precision"] / (num_samples * 3)
            avg_recall = aggregated_rouge_scores[rouge_type]["recall"] / (num_samples * 3)
            avg_fmeasure = aggregated_rouge_scores[rouge_type]["fmeasure"] / (num_samples * 3)
            
            # Log the average precision, recall, and F1 score for each ROUGE metric
            self.log(f'{rouge_type}_avg_precision', avg_precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{rouge_type}_avg_recall', avg_recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{rouge_type}_avg_fmeasure', avg_fmeasure, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Clear the list of outputs to prepare for the next validation epoch
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
        outputs = self.forward(**batch)
        test_loss = outputs.loss
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Generate text using the model for the test batch.
        generated_texts = self.generate_text(batch['input_ids'], batch.get('attention_mask'))
            
        # Decode the actual labels from the batch to get the ground truth text.
        decoded_targets = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['labels']]

        # Calculate custom metrics using the _calculate_metrics method.
        self._calculate_metrics(generated_texts, decoded_targets)

    def _calculate_metrics(self, generated_texts, decoded_targets):
        """
        Calculates and logs the BLEU and ROUGE metrics for the given batch of data.

        Args:
            generated_texts (list of str): The texts generated by the model.
            decoded_targets (list of str): The reference texts for comparison.

        """
        try:
            for gen, ref in zip(generated_texts, decoded_targets):
                # Calculate BLEU score
                bleu_score = sentence_bleu([ref.split()], gen.split())
                self.log('bleu', bleu_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

                # Calculate ROUGE scores
                rouge_scores = self.rouge_scorer.score(ref, gen)
                for rouge_type, scores in rouge_scores.items():
                    self.log(f'{rouge_type}_precision', scores.precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log(f'{rouge_type}_recall', scores.recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                    self.log(f'{rouge_type}_fmeasure', scores.fmeasure, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        except Exception as e:
            # Log an error message and a custom metric in case of an exception during metric calculation.
            logger.error(f"An error occurred during metric calculation: {e}")
            self.log('metric_calculation_error', 1, on_step=False, on_epoch=True, logger=True)

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

    def generate_text(self, input_ids, attention_mask=None, max_length=512):
        """
        Generates text sequences from the provided input_ids using the model. This method is typically used during 
        the validation and testing phases to transform model predictions (in the form of token IDs) into human-readable 
        text. The method leverages the model's generate function, which ensures that the text generation process adheres 
        to the nuances of the specific model architecture (e.g., handling of attention masks, managing token types, etc.).

        Args:
            input_ids (torch.Tensor): A tensor containing the token IDs for the input text. These IDs are numerical 
                                        representations of the input text as understood by the model.
            attention_mask (torch.Tensor, optional): A binary tensor indicating the position of padded indices so 
                                                        that the model does not attend to them. Defaults to None.
            max_length (int, optional): The maximum length of the sequence to be generated. Defaults to 512.

        Returns:
            list of str: A list containing the generated text sequences. Each sequence in the list corresponds to the 
                            generated text for a single input in the input_ids tensor.
        """
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
        generated_texts = [self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for generated_id in generated_ids]
        return generated_texts