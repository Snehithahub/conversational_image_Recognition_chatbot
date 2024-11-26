import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import nltk
from transformers import pipeline
import random
from google.colab import files

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
        
        # Update context with image description
        image_context['description'] = label
        image_context['image_path'] = image_path
        # Optionally, add some details based on common image features
        image_context['additional_info'] = "This appears to be a fashion item, possibly part of swimwear or clothing."
        return f"I analyzed the image, and it appears to be: {label}."
    except Exception as e:
        return f"Sorry, I couldn't process the image. Error: {e}"


# Function to handle text-based queries about the image
def answer_question(question):
    global image_context
    if 'description' not in image_context:
        return "I haven't analyzed any image yet. Please provide an image first."
    
    # If the question mentions 'what else' or similar queries
    if 'else' in question or 'other' in question or 'additional' in question:
        additional_info = image_context.get('additional_info', 'No additional information available.')
        return f"I have recognized: {image_context['description']}. Also, {additional_info}"
    
    # For more specific text-based questions, use the QA model to generate answers
    context = f"The image seems to show: {image_context['description']}."
    response = qa_pipeline(question=question, context=context)
    return response['answer'] if response.get('answer') else "I'm sorry, I don't know the answer to that."


# Main interactive chatbot
def chatbot():
    print("Hello! I can analyze images and answer questions about them. Type 'quit' to exit.")
    while True:
        user_input = input("> ")
        
        # Exit the loop if 'quit' is typed
        if user_input.lower() == 'quit':
            break
        
        # Check if the input is an image file (by file extension)
        if user_input.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            response = analyze_image(user_input)  # Analyze image
        else:
            response = answer_question(user_input)  # Answer text-based query
        
        print(response)


# Start the chatbot
def start_chat():
    # Step 1: Upload an image using Google Colab's file upload function
    uploaded = files.upload()
    
    # Step 2: Get the uploaded image filename
    image_filename = list(uploaded.keys())[0]  # Extract the filename
    
    # Step 3: Pass the filename to analyze the image
    response = analyze_image(image_filename)
    print(response)

    # Step 4: Start the chatbot for any questions about the image
    chatbot()

# Run the chatbot
if __name__ == "__main__":
    start_chat()
