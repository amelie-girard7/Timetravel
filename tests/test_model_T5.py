# tests/test_model_T5.py
import unittest
from src.models.model_T5 import FlanT5FineTuner
from transformers import T5Tokenizer
import torch
from unittest.mock import MagicMock

class TestFlanT5FineTuner(unittest.TestCase):
    
    def setUp(self):
        # Initialize things you'll need for the tests (like model, tokenizer)
        # Note: Use a dummy model or a very small version for testing purposes
        self.model = FlanT5FineTuner('google/t5-small')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model.tokenizer = self.tokenizer  # Assume tokenizer is part of the model

        # Mock the actual call to the model to avoid heavy computations during testing
        self.model.forward = MagicMock(return_value=torch.rand((1, 1)))

    def test_forward_pass(self):
        # Test the forward pass of the model
        input_text = "Rewrite the story based on: She likes grapes, but she went to the store."
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        
        # Perform forward pass
        output = self.model(input_ids=input_ids)
        
        # Check if output is not None
        self.assertIsNotNone(output)
        
    def test_training_step(self):
        # Test the training_step method
        batch = {'input_ids': torch.rand((1, 512)), 'attention_mask': torch.rand((1, 512)), 'labels': torch.rand((1, 512))}
        loss = self.model.training_step(batch, 0)
        
        # Check if loss is a tensor
        self.assertIsInstance(loss, torch.Tensor)
        
    def test_validation_step(self):
        # Test the validation_step method
        batch = {'input_ids': torch.rand((1, 512)), 'attention_mask': torch.rand((1, 512)), 'labels': torch.rand((1, 512))}
        self.model.validation_step(batch, 0)
        
        # Test if validation_step runs without errors (mainly for coverage)
        self.assertTrue(True)

    def test_test_step(self):
        # Test the test_step method
        batch = {'input_ids': torch.rand((1, 512)), 'attention_mask': torch.rand((1, 512)), 'labels': torch.rand((1, 512))}
        self.model.test_step(batch, 0)
        
        # Test if test_step runs without errors (mainly for coverage)
        self.assertTrue(True)

    def test_configure_optimizers(self):
        # Test the configure_optimizers method
        optimizer = self.model.configure_optimizers()
        
        # Check if the method returns an optimizer
        self.assertIsNotNone(optimizer)

    def test_generate_text(self):
        # Test the generate_text method
        input_ids = torch.randint(low=0, high=200, size=(1, 512))
        generated_texts = self.model.generate_text(input_ids)
        
        # Check if generated_texts is a list
        self.assertIsInstance(generated_texts, list)
        
        # Check if generated_texts is not empty
        self.assertTrue(len(generated_texts) > 0)

if __name__ == '__main__':
    unittest.main()
