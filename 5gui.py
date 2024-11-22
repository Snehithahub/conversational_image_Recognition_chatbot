import os
import tkinter as tk
from tkinter import filedialog, Text, Scrollbar, messagebox
import torch
from torchvision import models, transforms
from PIL import Image, ImageTk
from transformers import pipeline
import random
import nltk

# Download NLTK resources
nltk.download('punkt')

# Load pre-trained ResNet50 model for image recognition
image_model = models.resnet50(pretrained=True)
image_model.eval()

# Load ImageNet labels
import requests
imagenet_labels = requests.get(
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
).json()

# Load pre-trained Hugging Face question-answering model
qa_pipeline = pipeline("question-answering")

# Global context for conversation
context = {"mode": "text", "image_description": None}

# Image preprocessing pipeline
image_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Function to analyze the image
def analyze_image(image_path):
    try:
        global context
        image = Image.open(image_path).convert('RGB')
        input_tensor = image_preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = image_model(input_tensor)
        _, predicted_class = output.max(1)
        label = imagenet_labels[predicted_class.item()]
        context['mode'] = 'image'
        context['image_description'] = label
        return f"I analyzed the image, and it seems to be: {label}."
    except Exception as e:
        return f"Error processing the image: {str(e)}"


# Function to generate a response to user queries
def generate_response(user_input):
    global context
    if context['mode'] == 'image' and context['image_description']:
        # Respond based on image context
        response = qa_pipeline(question=user_input, context=f"The image shows: {context['image_description']}.")
        return response['answer']
    else:
        # General fallback responses
        return random.choice([
            "I'm here to help! What can I do for you?",
            "Tell me more so I can assist you better.",
            "I'm not sure I understand. Can you clarify?",
        ])


# Function to handle user input (text or image)
def handle_input(event=None):
    user_input = entry_box.get()
    if not user_input.strip():
        return

    # Display user input
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, f"You: {user_input}\n")
    chat_log.config(state=tk.DISABLED)
    entry_box.delete(0, tk.END)

    # Generate and display response
    response = generate_response(user_input)
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, f"Bot: {response}\n")
    chat_log.config(state=tk.DISABLED)
    chat_log.yview(tk.END)


# Function to handle image upload
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    if not file_path:
        return

    # Analyze the uploaded image
    response = analyze_image(file_path)

    # Display the image in the chat
    try:
        img = Image.open(file_path).resize((150, 150))
        img = ImageTk.PhotoImage(img)
        chat_log.image_create(tk.END, image=img)
        chat_log.image = img  # Keep a reference to avoid garbage collection
    except Exception:
        pass

    # Display the analysis result
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, f"\nBot: {response}\n")
    chat_log.config(state=tk.DISABLED)
    chat_log.yview(tk.END)


# GUI Setup
root = tk.Tk()
root.title("Chatbot with Image Analysis")

# Chat display area
chat_frame = tk.Frame(root)
scrollbar = Scrollbar(chat_frame)
chat_log = Text(chat_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, state=tk.DISABLED, height=20, width=50)
scrollbar.config(command=chat_log.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
chat_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
chat_frame.pack(padx=10, pady=10)

# Input area
input_frame = tk.Frame(root)
entry_box = tk.Entry(input_frame, width=40)
entry_box.pack(side=tk.LEFT, padx=5)
entry_box.bind("<Return>", handle_input)
send_button = tk.Button(input_frame, text="Send", command=handle_input)
send_button.pack(side=tk.LEFT, padx=5)
input_frame.pack(padx=10, pady=5)

# Image upload button
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=5)

# Run the GUI
root.mainloop()
