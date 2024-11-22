import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from transformers import pipeline
import random
import nltk

# Download required nltk resources
nltk.download('punkt')

# Image preprocessing pipeline
image_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pre-trained ResNet50 model for image recognition
image_model = models.resnet50(pretrained=True)
image_model.eval()

# Load ImageNet labels
import requests
imagenet_labels = requests.get(
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
).json()

# Load pre-trained question-answering pipeline
qa_pipeline = pipeline("question-answering")

# Context management for chatbot
context = {"mode": "text", "image_description": None}

# Function to classify an image
def analyze_image(image_path):
    try:
        # Preprocess and classify the image
        image = Image.open(image_path).convert('RGB')
        input_tensor = image_preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = image_model(input_tensor)
        _, predicted_class = output.max(1)
        label = imagenet_labels[predicted_class.item()]
        
        # Update context with image description
        context['mode'] = 'image'
        context['image_description'] = label
        return f"I analyzed the image, and it seems to be: {label}."
    except Exception as e:
        return f"Sorry, I couldn't process the image. Error: {e}"

# Function to handle text-based conversation
def text_response(user_input):
    if context['mode'] == 'image' and context['image_description']:
        # If in image mode, generate responses about the image
        response = qa_pipeline(question=user_input, context=f"The image shows: {context['image_description']}.")
        return response['answer']
    else:
        # General fallback response for chat mode
        return random.choice([
            "I'm here to help! What can I do for you?",
            "Tell me more so I can assist you better.",
            "I'm not sure I understand. Can you clarify?",
        ])

# Unified chatbot function
def chatbot():
    print("Hi! I'm a chatbot. I can answer your questions or analyze images. Type 'quit' to exit.")
    while True:
        user_input = input("> ")
        if user_input.lower() == "quit":
            break

        # Check if input is an image path
        if user_input.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            response = analyze_image(user_input)
        else:
            response = text_response(user_input)

        print(response)

# Run the chatbot
if __name__ == "__main__":
    chatbot()
