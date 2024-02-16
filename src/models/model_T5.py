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
    
    
    
    def forward(self, input_ids, attention_mask, labels):
        """
        Performs the forward pass of the model.b4         
        Args:-992            premise (Tensor): Tokenized tensor for the story premises.
            initial (Tensor): Tokenized tensor for the initial states of the stories.
            original_ending (Tensor): Tokenized tensor for the original endings of the stories.
            counterfactual (Tensor): Tokenized tensor for the counterfactual (alternative scenarios) of the stories.
            labels (Tensor, optional): Tokenized tensor for the edited endings, serving as labels for training. Default is None.
            attention_mask (Tensor, optional): Tensor indicating which tokens should be attended to, and which should not.
        
        Returns:
            The output from the T5 model, which includes loss when labels are provided, and logits otherwise.
        """
        print("--forward pass--")
        
        if labels is not None:
            print(f"Labels shape: {labels.shape}")
        
        # Pass the concatenated input_ids, attention_mask, and labels (if provided) to the model.
        # The T5 model expects input_ids and attention_mask for processing.
        # If labels are provided (during training), the model will also return the loss
        # which can be used to update the model's weights.
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        # If labels were provided, the model's output will include loss for training.
        # During inference (no labels), the model generates logits from which we can derive predictions.
        if labels is not None:
            print("Loss from model output:", output.loss.item())
        else:
            print("Model output generated without calculating loss (inference mode).")
        
        return output



    def training_step(self, batch, batch_idx):
        """  
        Defines the training logic for a single batch, where a forward pass is performed and the loss is calculated.

        Args:
            batch (dict): The batch of data provided by the DataLoader.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        print("--training_step --")    
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
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
        print("-- validation_step --")   
        # Forward pass to compute loss and model outputs
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        val_loss = outputs.loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Generate text predictions from the model using the individual components
        generated_texts = self.generate_text(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )

        # Decode the labels (ground truth edited ending) from the batch for comparison with the model's generated text.
        edited_endings = batch['edited_ending']

        # Store output information for metric calculation at the end of the epoch
        output = {
            'generated': generated_texts,
            'edited_endings': edited_endings,
            # Add other story components for later use in metric calculations
            'premises': batch['premise'],
            'counterfactuals': batch['counterfactual'],
            'original_endings': batch['original_ending'],
            'initials': batch['initial'],
        }
        self.current_val_step_outputs.append(output)


    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to calculate and log metrics.
        """
        print("-- on_validation_epoch_end --")   
        # Initialize variables to store aggregated metrics
        aggregated_bleu_scores = 0
        aggregated_rouge_scores = {rouge_type: {"precision": 0, "recall": 0, "fmeasure": 0} for rouge_type in self.rouge_types}
        num_samples = len(self.current_val_step_outputs)  # Number of samples to average metrics over

        # Loop through each output from the validation steps
        for output in self.current_val_step_outputs:
            generated_texts = output['generated']
            edited_endings = output['edited_endings']
            
            # Calculate metrics for each generated-reference pair in the batch
            # TODO: Change to Sacrebleu corpus level, remove for loop
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
    
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def calculate_metrics(self, generated_texts, edited_endings):
        """
        Calculates and logs metrics for the generated texts against the edited endings.
        """
        print("-- calculate_metrics --") 
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
        print("-- configure_optimizers --") 
        
        #lr = CONFIG.get("learning_rate", 2e-5)  # Fetch the learning rate from CONFIG with a default
        #return torch.optim.AdamW(self.model.parameters(), lr=lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return optimizer

    def generate_text(self, input_ids, attention_mask):
        """
        Generates text sequences from the provided input components using the model.
        """
        print("-- generate_text --") 

        # Generate a tensor of token IDs based on the input_ids and attention_mask
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
        
        # Decode the generated token IDs back into human-readable text
        generated_texts = [self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for generated_id in generated_ids]
        
        return generated_texts