import torch
import time
from torchinfo import summary
from pytorch_memlab import LineProfiler

from Kosmos.torchscale.torchscale.component.multihead_attention import MultiheadAttention

def test_multihead_attention():
    batch_size = 64
    d_model = 512
    num_heads = 8

    multihead_attention = MultiheadAttention(
        embed_dim=d_model, 
        num_heads=num_heads, 
        dropout=0.1, 
        flash_attn=True
    )

    # Choose a set of sequence lengths to test
    sequence_lengths = [2**n for n in range(10, 16)]  # 1024, 2048, ..., 32768

    for seq_len in sequence_lengths:
        print(f'Testing sequence length: {seq_len}')

        # Create some dummy data
        query = torch.rand(batch_size, seq_len, d_model)
        key = torch.rand(batch_size, seq_len, d_model)
        value = torch.rand(batch_size, seq_len, d_model)

        # Time the forward pass
        start_time = time.time()
        output = multihead_attention(query, key, value)
        end_time = time.time()
        print(f'Time taken for forward pass: {end_time - start_time} seconds')

        # Compute the FLOPs
        flops = summary(multihead_attention, input_size=(batch_size, seq_len, d_model))
        print(f'FLOPs: {flops.total_ops}')

        # Compute the memory usage
        profiler = LineProfiler()
        profiler.add_function(multihead_attention.forward)
        profiler.add_module(multihead_attention)
        profiler.run('output = multihead_attention(query, key, value)')
        print('Memory usage: ', profiler.display())
        
test_multihead_attention()
