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
        # Initialize the model, tokenizer, and rouge scorer.
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Initialize RougeScorer with desired metrics
        rouge_types = ['rouge1', 'rougeL']  # Specify the types of ROUGE metrics you want
        self.rouge_scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        self.rouge_types = rouge_types  # Store the types of ROUGE metrics for later reference

        # Initialize a list to store outputs for each validation step.
        self.current_val_step_outputs = [] 

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
        Executes the validation logic for a single batch during the model's validation phase. This method is 
        automatically called by PyTorch Lightning for each batch of the validation data loader.

        The method performs a forward pass with the given batch, calculates the loss, and logs it for monitoring. 
        Additionally, it utilizes the model's capability to generate text based on the input_ids from the batch. 
        The generated texts and the actual labels (ground truth) are stored for later metric calculation at the 
        end of the validation epoch.

        Args:
            batch (dict): A dictionary containing the input_ids, attention_mask, and labels for the batch. 
                        These are automatically passed by the DataLoader.
            batch_idx (int): The index of the current batch. While it's provided by PyTorch Lightning, it's 
                            not used in this method but can be useful for logging or conditional processing.
        """
        # Perform a forward pass with the model using the input from the batch. This calculates the loss among other things.
        outputs = self.forward(**batch)
        val_loss = outputs.loss
        # Log the validation loss to track model performance. This is useful for monitoring and early stopping if needed.
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Use the model to generate text based on the input_ids. This text is used to evaluate the model's performance on validation data.
        generated_texts = self.generate_text(batch['input_ids'], batch.get('attention_mask'))
        # Decode the labels (ground truth) from the batch for comparison with the model's generated text.
        references = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['labels']]
        # Save the outputs as an instance attribute. 
        # This data will be used for calculating aggregated validation metrics at the end of the epoch.
        output = {'generated': generated_texts, 'references': references}
        self.current_val_step_outputs.append(output)  # Save the outputs for each validation step

 
    def on_validation_epoch_end(self):
        """
        Called at the end of a validation epoch to aggregate and calculate metrics over the entire validation dataset.
        
        This method collects all the generated texts and their corresponding references from the validation steps,
        calculates ROUGE scores at the dataset level, and logs these metrics. It ensures that metrics like ROUGE
        are computed over the whole dataset, providing a more accurate and holistic measure of the model's performance.
        """
        # Aggregate all generated texts and references from the entire validation epoch
        all_generated = [output['generated'] for output in self.current_val_step_outputs]
        all_references = [output['references'] for output in self.current_val_step_outputs]
       
        # Debug print
        print(f"Type of all_generated: {type(all_generated)}, First element type: {type(all_generated[0])}")
        print(f"Type of all_references: {type(all_references)}, First element type: {type(all_references[0])}")
        
        # Check if the first element is a list, and if so, join the tokens to form a sentence
        if all_generated and isinstance(all_generated[0], list):
            all_generated = [' '.join(gen) for gen in all_generated]
        if all_references and isinstance(all_references[0], list):
            all_references = [' '.join(ref) for ref in all_references]
        
        # Initialize a dictionary to store the aggregated ROUGE scores
        aggregated_rouge_scores = {rouge_type: {"precision": 0, "recall": 0, "fmeasure": 0} for rouge_type in self.rouge_types}
        
        # Calculate the ROUGE scores for each pair of generated and reference texts
        for ref, gen in zip(all_references, all_generated):
            rouge_scores = self.rouge_scorer.score(ref, gen)
            for rouge_type, scores in rouge_scores.items():
                aggregated_rouge_scores[rouge_type]["precision"] += scores.precision
                aggregated_rouge_scores[rouge_type]["recall"] += scores.recall
                aggregated_rouge_scores[rouge_type]["fmeasure"] += scores.fmeasure
        
        # Calculate the average of the ROUGE scores
        num_samples = len(all_generated)
        for rouge_type, scores in aggregated_rouge_scores.items():
            aggregated_rouge_scores[rouge_type]["precision"] /= num_samples
            aggregated_rouge_scores[rouge_type]["recall"] /= num_samples
            aggregated_rouge_scores[rouge_type]["fmeasure"] /= num_samples
            # Log the average scores
            self.log(f'rouge_{rouge_type}', scores["fmeasure"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Clear the val_step_outputs list to prepare for the next validation epoch.
        self.current_val_step_outputs = []


    def test_step(self, batch, batch_idx):
        """
        Called during the testing loop to perform a forward pass with a batch from the test set and calculate the loss.
        
        This method also generates texts using the model based on the input_ids from the batch. It's structured to allow
        for the potential calculation of custom metrics by calling the _calculate_metrics method, which can be customized
        to compute metrics specific to the testing dataset. 
        
        Args:
            batch (dict): The batch of data provided by the DataLoader. It contains input_ids, attention_mask, and labels.
            batch_idx (int): The index of the current batch. This parameter is not directly used in this method but is
                            included to match the expected method signature by PyTorch Lightning.      
        """
        # Perform a forward pass and calculate the loss for the test batch.
        outputs = self.forward(**batch)
        test_loss = outputs.loss
        # Log the test loss.
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Generate text using the model for the test batch.
        generated_texts = self.generate_text(batch['input_ids'], batch.get('attention_mask'))
        
        # Calculate custom metrics if needed using the _calculate_metrics method.
        self._calculate_metrics(batch, outputs)

    def _calculate_metrics(self, batch, outputs):
        """
        Calculates and logs the BLEU and ROUGE metrics for the given batch of data. This method is designed to be used 
        during validation and testing to quantitatively evaluate the model's performance by comparing the generated 
        text against the ground truth.

        Args:
            batch (dict): The batch of data provided by the DataLoader. It contains input_ids, attention_mask, and labels, 
                        which are used to generate texts and compare with the ground truth.
            outputs (Seq2SeqLMOutput): The outputs from the model's forward pass. It contains logits from which the 
                                    generated texts are obtained.

        This method calculates the BLEU score, which is a measure of the similarity between the generated text and the 
        reference text. It also calculates the ROUGE score, specifically ROUGE-1 and ROUGE-L, which are measures of the 
        quality of the generated text based on precision, recall, and F1 score. These metrics are logged for monitoring 
        and analysis purposes.
        """
        try:
            # Extract logits from the model's outputs and decode them to get the predicted text.
            logits = outputs.logits
            decoded_preds = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in logits]
            # Decode the actual labels from the batch to get the ground truth text.
            decoded_targets = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['labels']]

            # Calculate BLEU score, which is a measure of the similarity between the generated text and the reference text.
            bleu_score = sentence_bleu([decoded_targets], decoded_preds)
            # Log the BLEU score for the current step/epoch.
            self.log('bleu', bleu_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            # Calculate ROUGE scores, which measure the quality of the generated text based on precision, recall, and F1 score.
            rouge_scores = {key: self.rouge_scorer.score(decoded_targets, decoded_preds) for key in self.rouge_scorer.metrics}
            # Log each ROUGE score for the current step/epoch.
            for key, scores in rouge_scores.items():
                self.log(f'rouge_{key}', scores.fmeasure, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
        lr = CONFIG.get("learning_rate", 2e-5)  # Fetch the learning rate from CONFIG with a default
        return torch.optim.AdamW(self.model.parameters(), lr=lr)

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
        # Use the model to generate a tensor of token IDs based on the input_ids and attention_mask.
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
        
        # Decode the generated token IDs back into human-readable text, ensuring that special tokens are ignored and 
        # extra spaces introduced during tokenization are cleaned up.
        generated_texts = [self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for generated_id in generated_ids]
        
        return generated_texts

