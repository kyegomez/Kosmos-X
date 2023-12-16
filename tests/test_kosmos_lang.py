import pytest
import torch
from kosmosx.model import KosmosLanguage


def test_kosmoslanguage_initialization():
    model = KosmosLanguage()
    assert isinstance(model, KosmosLanguage)


def test_kosmoslanguage_forward():
    model = KosmosLanguage()
    x = torch.randint(0, 32002, (1, 50), dtype=torch.long)
    output = model.forward(x)
    assert output.shape == (1, 32002)


@pytest.mark.parametrize(
    "vocab_size,dim,depth,ffn_dim,dropout,multiway,decoder_heads,activation_fn,subln,alibi_pos_bias,alibi_num_heads,xpos_rel_pos,max_rel_pos",
    [
        (
            10000,
            2048,
            24,
            8192,
            0.1,
            True,
            32,
            "gelu",
            True,
            True,
            16,
            True,
            2048,
        ),
        (
            5000,
            1024,
            12,
            4096,
            0.05,
            False,
            16,
            "relu",
            False,
            False,
            8,
            False,
            1024,
        ),
        (
            20000,
            4096,
            48,
            16384,
            0.2,
            True,
            64,
            "swish",
            True,
            True,
            32,
            True,
            4096,
        ),
    ],
)
def test_kosmoslanguage_with_different_parameters(
    vocab_size,
    dim,
    depth,
    ffn_dim,
    dropout,
    multiway,
    decoder_heads,
    activation_fn,
    subln,
    alibi_pos_bias,
    alibi_num_heads,
    xpos_rel_pos,
    max_rel_pos,
):
    model = KosmosLanguage(
        vocab_size,
        dim,
        depth,
        ffn_dim,
        dropout,
        multiway,
        decoder_heads,
        activation_fn,
        subln,
        alibi_pos_bias,
        alibi_num_heads,
        xpos_rel_pos,
        max_rel_pos,
    )
    assert isinstance(model, KosmosLanguage)


def test_kosmoslanguage_forward_with_different_input_sizes():
    model = KosmosLanguage()
    for i in range(1, 6):
        x = torch.randint(0, 32002, (i, 50), dtype=torch.long)
        output = model.forward(x)
        assert output.shape == (i, 32002)


# Additional tests for output
def test_kosmoslanguage_output_values():
    model = KosmosLanguage()
    x = torch.randint(0, 32002, (1, 50), dtype=torch.long)
    output = model.forward(x)
    assert (output >= 0).all() and (output <= 1).all()


def test_kosmoslanguage_output_with_zero_input():
    model = KosmosLanguage()
    x = torch.zeros((1, 50), dtype=torch.long)
    output = model.forward(x)
    assert (output == 0).all()
