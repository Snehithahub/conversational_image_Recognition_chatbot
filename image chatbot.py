import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import nltk
from transformers import pipeline
import random

# Download nltk resources
nltk.download('punkt')

# Define image processing pipeline
image_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load a pre-trained image recognition model (e.g., ResNet50)
image_model = models.resnet50(pretrained=True)
image_model.eval()

# Define labels for classification (using ImageNet classes as an example)
# You can download ImageNet labels (https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json)
import requests
imagenet_labels = requests.get("https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json").json()

# Load a pre-trained question-answering model (e.g., using Hugging Face's pipeline)
qa_pipeline = pipeline("question-answering")

# Store extracted image details for answering questions
image_context = {}


# Function to classify an image and store context
def analyze_image(image_path):
    global image_context
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        input_tensor = image_preprocess(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = image_model(input_tensor)
        _, predicted_class = output.max(1)
        label = imagenet_labels[predicted_class.item()]
        
        # Update context
        image_context['description'] = label
        image_context['image_path'] = image_path
        return f"I analyzed the image, and it appears to be: {label}."
    except Exception as e:
        return f"Sorry, I couldn't process the image. Error: {e}"


# Function to handle text-based queries about the image
def answer_question(question):
    global image_context
    if 'description' not in image_context:
        return "I haven't analyzed any image yet. Please provide an image first."
    
    # Generate context for QA model
    context = f"The image seems to show: {image_context['description']}."
    response = qa_pipeline(question=question, context=context)
    return response['answer']


# Main interactive chatbot
def chatbot():
    print("Hello! I can analyze images and answer questions about them. Type 'quit' to exit.")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'quit':
            break
        
        # Check if the input is an image file
        if user_input.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            response = analyze_image(user_input)
        else:
            response = answer_question(user_input)
        
        print(response)


# Start the chatbot
if __name__ == "__main__":
    chatbot()
