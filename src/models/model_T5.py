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
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        The forward method is called during training and validation steps.
        It performs a forward pass through the model, returning a Seq2SeqLMOutput object.
        
        If labels are provided, the model returns the loss, which is the Cross-Entropy Loss
        between the model predictions and the actual labels, facilitating training.
        """
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        """
        Defines the training logic for a single batch.
        The method performs a forward pass and uses the returned loss to perform backpropagation.
        """
        outputs = self.forward(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation logic for a single batch.
        It calculates the loss for validation data and can also generate text for further analysis.
        """
        outputs = self.forward(**batch)
        val_loss = outputs.loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Generate text based on the input
        generated_texts = self.generate_text(batch['input_ids'], batch.get('attention_mask'))
        # Log the first generated text for inspection (optional)
        if batch_idx == 0:  # log the first batch's first generated text
            logger.info(f"Sample Generated text: {generated_texts[0]}")

        # Calculate custom metrics
        self._calculate_metrics(batch, outputs)

    def test_step(self, batch, batch_idx):
        """
        Defines the test logic for a single batch.
        Similar to the validation step, it calculates the loss and can be used to generate text.
        """
        outputs = self.forward(**batch)
        test_loss = outputs.loss
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Generate text and calculate metrics if needed
        self._calculate_metrics(batch, outputs)

    def _calculate_metrics(self, batch, outputs):
        """
        Calculates and logs the BLEU and ROUGE metrics for the given batch.
        This method is used during validation and testing to measure the model's performance.
        """
        try:
            logits = outputs.logits
            decoded_preds = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in logits]
            decoded_targets = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['labels']]

            # Calculating BLEU score, note that sentence_bleu expects a list of reference sentences
            bleu_score = sentence_bleu([decoded_targets], decoded_preds)
            self.log('bleu', bleu_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            # Calculating ROUGE scores
            rouge_scores = {key: self.rouge_scorer.score(decoded_targets, decoded_preds) for key in self.rouge_scorer.metrics}
            for key, scores in rouge_scores.items():
                self.log(f'rouge_{key}', scores.fmeasure, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        except Exception as e:
            logger.error(f"An error occurred during metric calculation: {e}")
            # If you want to log a metric to indicate an error occurred, log a numerical value, like:
            self.log('metric_calculation_error', 1, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
        The optimizer is responsible for updating the model's weights to minimize the loss during training.
        
        Returns:
            The optimizer to be used for training the model.
        """
        lr = CONFIG.get("learning_rate", 2e-5)  # Fetch the learning rate from CONFIG with a default
        return torch.optim.AdamW(self.model.parameters(), lr=lr)

    def generate_text(self, input_ids, attention_mask=None, max_length=512):
        """
        Generate text using the model.
        This method is used during validation and testing to see the model's generated output.
        
        Parameters:
            input_ids (torch.Tensor): Tensor of token ids to be fed to the model.
            attention_mask (torch.Tensor): Tensor of attention mask to be fed to the model.
            max_length (int): Maximum length of the generated text.
        
        Returns:
            list: The generated texts.
        """
        # Use the model to generate token ids
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
        
        # Decode the generated token ids to texts
        generated_texts = [self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for generated_id in generated_ids]
        
        return generated_texts
