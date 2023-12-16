import pytest
import torch
from kosmosx.model import Kosmos, KosmosTokenizer
from zeta import StableAdamWUnfused


@pytest.fixture
def setup():
    model = Kosmos()
    tokenizer = KosmosTokenizer()
    optimizer = StableAdamWUnfused(model.parameters())
    loss_function = torch.nn.CrossEntropyLoss()
    input_text = ["<image>", "</image>"]
    input_images = torch.randn(1, 3, 224, 224)
    return model, tokenizer, optimizer, loss_function, input_text, input_images


def test_forward_pass(setup):
    model, tokenizer, _, _, input_text, input_images = setup
    tokenized_input = tokenizer.tokenize_texts(input_text)
    output = model(*tokenized_input, input_images)
    assert output.shape == (1, 1024, 64007)  # verify output shape


def test_backward_pass(setup):
    model, tokenizer, _, loss_function, input_text, input_images = setup
    model.zero_grad()
    tokenized_input = tokenizer.tokenize_texts(input_text)
    output = model(*tokenized_input, input_images)
    loss = loss_function(output.squeeze(), tokenized_input[0])
    loss.backward()
    for name, parameter in model.named_parameters():
        assert not torch.isnan(
            parameter.grad
        ).any(), f"Gradient for {name} contains NaNs"
        assert not torch.isinf(
            parameter.grad
        ).any(), f"Gradient for {name} contains Infs"


def test_optimizer_step(setup):
    model, tokenizer, optimizer, loss_function, input_text, input_images = setup
    initial_params = [param.clone() for param in model.parameters()]
    tokenized_input = tokenizer.tokenize_texts(input_text)
    output = model(*tokenized_input, input_images)
    loss = loss_function(output.squeeze(), tokenized_input[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for initial_param, updated_param in zip(initial_params, model.parameters()):
        assert not torch.equal(initial_param, updated_param)
