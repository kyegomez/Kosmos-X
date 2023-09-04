import torch
from kosmosx.model import Kosmos

# Create a sample text token tensor with dtype torch.long
text_tokens = torch.randint(0, 32002, (1, 50), dtype=torch.long)

# Create a sample image tensor
images = torch.randn(1, 3, 224, 224)
images = images.long()

# Instantiate the model
model = Kosmos()

# Pass the sample tensors to the model's forward function
output = model.forward(
    text_tokens=text_tokens,
    images=images
)

# Print the output from the model
print(f"Output: {output}")