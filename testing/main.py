import unittest
import torch
from kosmosx.model import Kosmos, KosmosTokenizer
from kosmosx.utils.stable_adamw import StableAdamWUnfused

class KosmosTest(unittest.TestCase):

    def setUp(self):
        self.model = Kosmos()
        self.tokenizer = KosmosTokenizer()
        self.optimizer = StableAdamWUnfused(self.model.parameters())
        
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.input_text = ["<image>", "</image>"]
        self.input_images = torch.randn(1, 3, 224, 224)

    def test_forward_pass(self):
        tokenized_input = self.tokenizer.tokenize_texts(self.input_text)
        output = self.model(*tokenized_input, self.input_images)
        self.assertEqual(output.shape, (1, 1024, 64007)) #verify output shape

    def test_backward_pass(self):
        self.optimizer.zero_grad()
        tokenized_input = self.tokenizer.tokenize_texts(self.input_text)
        output = self.model(*tokenized_input, self.input_images)
        loss = self.loss_function(output.squeeze(), tokenized_input[0])

        loss.backward()
        for name, parameter in self.model_parameters():
            self.assertFalse(torch.isnan(parameter.grad).any(), f"Gradient for {name} contains NaNs")
            self.assertFalse(torch.isinf(parameter.grad).any(), f"Gradient for {name} contains Infs")


    def test_optimizer_step(self):
        initial_params = [param.clone() for param in self.model.parameters()]
        tokenized_input = self.tokenizer.tokenize_texts(self.input_text)
        output = self.model(*tokenized_input, self.input_images)

        loss = self.loss_function(output.squeeze(), tokenized_input[0])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for initial_param, param in zip(initial_params, self.model.parameters()):
            self.assertFalse(torch.equal(initial_param, param), 'Model parameters did not change after an optimizer step')

    def test_data_loader(self):
        pass

    def test_lr_scheduling_rate(self):
        pass

    def test_hardware_compatibility(self):
        #implement a hward capabiloty test here
        pass

    def test_reproducibility(self):
        pass

if __name__ == "__main__":
    unittest.main()
    