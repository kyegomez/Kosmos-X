import torchvision.transforms as transforms
from ..model import Kosmos, KosmosTokenizer
from PIL import Image

#random text
text = "This is a sample text"


#laod a sample image
image_path = "galaxy-andromeda.jpeg"
image = Image.open(image_path)
transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
image = transform(image)
image = image.unsqueeze(0) #add batch dimension


#instantiate tokenzier and tokenize inputs
tokenizer = KosmosTokenizer()
tokenized_inputs = tokenizer.tokenize({"target_text": text, "image": image})



model = Kosmos()


#call the forward function and prunt the output
output = model.forward(
    text_tokens=tokenized_inputs["text_tokens"],
    images = tokenized_inputs["images"]
)

print(output)