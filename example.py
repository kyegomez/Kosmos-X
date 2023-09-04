import torch
from kosmosx.model import Kosmos

# Initialize the model
model = Kosmos()

# Create dummy data
batch_size = 4
dummy_images = torch.randn(batch_size, 3, 224, 224)  # Simulating random images
text_length = 100  # Arbitrary length for text tokens
dummy_text_tokens = torch.randint(0, 32002, (batch_size, text_length))  # Simulating random text tokens

# Pass the dummy data to the model
outputs = model(text_tokens=dummy_text_tokens, images=dummy_images)

print(outputs.shape)  # Check the shape of the outputs

