import torch
import open_clip
from PIL import Image

# Load model (correct order)
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)

tokenizer = open_clip.get_tokenizer('ViT-B-32')

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Image
image = preprocess(Image.open("dataset/images/sample.jpg")).unsqueeze(0).to(device)

# Text
text = tokenizer(["a photo of a dog"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

# Normalize
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Similarity
similarity = (image_features @ text_features.T).item()

print("Similarity score:", similarity)