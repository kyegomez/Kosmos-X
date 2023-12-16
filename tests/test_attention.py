import pytest
import torch
import time
from torchinfo import summary
from pytorch_memlab import LineProfiler

from zeta import MultiheadAttention


@pytest.fixture
def multihead_attention():
    d_model = 512
    num_heads = 8
    return MultiheadAttention(
        embed_dim=d_model, num_heads=num_heads, dropout=0.1, flash_attn=True
    )


def test_multihead_attention(multihead_attention):
    batch_size = 64
    d_model = 512

    # Choose a set of sequence lengths to test
    sequence_lengths = [2**n for n in range(10, 16)]  # 1024, 2048, ..., 32768

    for seq_len in sequence_lengths:
        # Create some dummy data
        query = torch.rand(batch_size, seq_len, d_model)
        key = torch.rand(batch_size, seq_len, d_model)
        value = torch.rand(batch_size, seq_len, d_model)

        # Time the forward pass
        start_time = time.time()
        output = multihead_attention(query, key, value)
        end_time = time.time()

        # Assert that output is not None
        assert output is not None

        # Assert that output has the correct shape
        assert output.shape == (batch_size, seq_len, d_model)

        # Compute the FLOPs
        flops = summary(
            multihead_attention, input_size=(batch_size, seq_len, d_model)
        )

        # Assert that FLOPs are not None
        assert flops.total_ops is not None

        # Compute the memory usage
        profiler = LineProfiler()
        profiler.add_function(multihead_attention.forward)
        profiler.add_module(multihead_attention)
        profiler.run("output = multihead_attention(query, key, value)")

        # Assert that memory usage is not None
        assert profiler.memory_usage is not None
