import torch
from kosmosx.model import KosmosLanguage

# Create a sample text token tensor with dtype torch.long
text = torch.randint(0, 32002, (1, 2048), dtype=torch.long)


# Instantiate the model
model = KosmosLanguage(
    vocab_size=32002,
    dim=2048,
)

# Pass the sample tensors to the model's forward function
output = model.forward(text)

# Print the output from the model
print(f"Output: {output}")
