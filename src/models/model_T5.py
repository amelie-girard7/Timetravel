import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pytorch_lightning as pl
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

class FlanT5FineTuner(pl.LightningModule):
    """
    PyTorch Lightning model for fine-tuning Flan-T5.
    """

    def __init__(self, model_path):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)  # Initialize Rouge scorer

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass of the model.
        """
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def _calculate_metrics(self, batch, outputs):
        """
        Calculate and log the BLEU and ROUGE metrics.
        """
        # Decode the predicted ids to texts
        decoded_preds = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs.predictions]
        decoded_targets = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['labels']]
        
        # Calculate BLEU
        bleu_score = sentence_bleu(decoded_targets, decoded_preds)
        self.log('bleu', bleu_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Calculate ROUGE
        rouge_score = self.rouge_scorer.score(decoded_targets, decoded_preds)
        for key in rouge_score:
            self.log(f'rouge_{key}', rouge_score[key].fmeasure, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        """
        Perform a training step.
        """
        outputs = self.forward(**batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step.
        """
        outputs = self.forward(**batch)
        val_loss = outputs.loss
        self.log('val_loss', val_loss, prog_bar=True)
        self._calculate_metrics(batch, outputs)  # Calculate metrics
        return val_loss

    def test_step(self, batch, batch_idx):
        """
        Perform a test step.
        """
        outputs = self.forward(**batch)
        test_loss = outputs.loss
        self.log('test_loss', test_loss, prog_bar=True)
        self._calculate_metrics(batch, outputs)  # Calculate metrics
        return test_loss

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
        """
        return torch.optim.AdamW(self.model.parameters(), lr=2e-5)