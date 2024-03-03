# src/models/model_T5.py
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pytorch_lightning as pl
#from nltk.translate.bleu_score import sentence_bleu
#from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from sacrebleu import corpus_bleu

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
        
        # Initialise sacre bleu
        self.sacre_bleu = BLEU()

        # Initialize RougeScorer with all types of ROUGE metrics
        """
        rouge_types = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        self.rouge_scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        self.rouge_types = rouge_types
        """
        
        # Initialize a list to store outputs for each validation step
        self.current_val_step_outputs = []
    
    
    
    def forward(self, input_ids, attention_mask, labels):
        """
        Performs the forward pass of the model.b4         
        Args:premise (Tensor): Tokenized tensor for the story premises.
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
        print(f"Validation loss: {val_loss.item()}")

        # Generate text predictions from the model using the individual components
        generated_texts = self.generate_text(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        # Decode the labels (ground truth edited ending) from the batch for comparison with the model's generated text.
        edited_endings = batch['edited_ending']

        #printing details for the first story in the batch
        print(f"--Premise: {batch['premise'][0]}")
        print(f"--Initial: {batch['initial'][0]}")
        print(f"--Counterfactual: {batch['counterfactual'][0]}")
        print(f"--Original Ending: {batch['original_ending'][0]}")
        print(f"--Edited Ending: {edited_endings[0]}")
        print(f"--Generated Text: {generated_texts[0]}")  


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
        print(f"Validation step {batch_idx} completed.")



    def on_validation_epoch_end(self):
        print("-- on_validation_epoch_end --")
        # Prepare lists to store generated texts and their corresponding references
        all_generated_texts = []
        all_edited_endings = []
        all_counterfactuals = []
        all_initials = []
        all_original_endings = []

        # Aggregate texts from the outputs
        for output in self.current_val_step_outputs:
            all_generated_texts.extend(output['generated'])
            all_edited_endings.extend(output['edited_endings'])
            all_counterfactuals.extend(output['counterfactuals'])
            all_initials.extend(output['initials'])
            all_original_endings.extend(output['original_endings'])
        print("Aggregated texts for BLEU score calculation.")

        # Convert lists for edited_endings to the format expected by SacreBLEU
        all_edited_endings_refs = [[ending] for ending in all_edited_endings]

        # Calculate and log BLEU scores for generated_text vs. story components
        comparisons = [
            ('bleu_prediction_edited', all_generated_texts, all_edited_endings_refs),
            ('bleu_prediction_cf', all_generated_texts, [[cf] for cf in all_counterfactuals]),
            ('bleu_prediction_initial', all_generated_texts, [[init] for init in all_initials]),
            ('bleu_prediction_original', all_generated_texts, [[orig] for orig in all_original_endings]),
        ]
        for label, hypotheses, references in comparisons:
            bleu_score = self.sacre_bleu.corpus_score(hypotheses, references)
            self.log(label, bleu_score.score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            print(f"{label}: {bleu_score.score}")

        # Calculate and log BLEU scores for edited_ending vs. other story components
        edited_comparisons = [
            ('bleu_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('bleu_edited_ending_initial', all_edited_endings, all_initials),
            ('bleu_edited_ending_original', all_edited_endings, all_original_endings),
        ]
        for label, edited_texts, component in edited_comparisons:
            bleu_score = self.sacre_bleu.corpus_score(edited_texts, [[comp] for comp in component])
            self.log(label, bleu_score.score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            print(f"{label}: {bleu_score.score}")

        # Clear the list of outputs for the next epoch
        self.current_val_step_outputs = []
        print("Validation epoch ended. Metrics logged.")


    def calculate_metrics(self, generated_texts, edited_endings):
        print("-- calculate_metrics --")
        # Initialize the BLEU scorer
        bleu_scorer = BLEU()

        # Ensure generated_texts and edited_endings are lists
        if not isinstance(generated_texts, list) or not isinstance(edited_endings, list):
            self.log("error", "Both generated_texts and edited_endings must be lists.")
            return
        if len(generated_texts) != len(edited_endings):
            self.log("error", "generated_texts and edited_endings must have the same number of elements.")
            return
        
        # Prepare the references and hypotheses for SacreBLEU
        # SacreBLEU expects a list of references for each hypothesis, hence the nested list comprehension
        references = [[ending] for ending in edited_endings]
        hypotheses = generated_texts
        
        # Calculate BLEU score
        try:
            # Note that SacreBLEU's corpus_bleu expects list of list of references and list of hypotheses
            bleu_score = bleu_scorer.corpus_score(hypotheses, references)
            aggregated_bleu_score = bleu_score.score
            self.log('avg_bleu', aggregated_bleu_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        except Exception as e:
            self.log("error", f"Error calculating BLEU score: {e}")
            
            
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