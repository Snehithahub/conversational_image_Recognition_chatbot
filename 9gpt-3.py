import streamlit as st
import openai
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# OpenAI API Key
openai.api_key = 'aask-proj-fJ2SD-Ne75Un8QaWrV3p2N2fuyuCqmUoW-7SbYmGxhps0NZBz2asKI8XY_shNXEokuptIAtZvNT3BlbkFJv9bvUTq6u9Yri8VARx-qy7EFdYBuaCp-qS4Fje_Zk0x_YzWs22TG3CF9sqfZ9w2dbDEB8zrPoA'  # Replace with your actual OpenAI API key

# Load CLIP model and processor for image captioning
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Function to generate text using GPT-3
def chat_with_gpt3(user_input):
    try:
        # Send a prompt to GPT-3 and get the response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can use "gpt-4" if you have access to it
            messages=[{"role": "user", "content": user_input}]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return str(e)

# Function to generate image caption using CLIP
def get_image_caption(image):
    # Prepare the image input for CLIP
    text_descriptions = [
        "a photo of a cat", "a photo of a dog", "a person walking on the beach", 
        "a beautiful landscape", "a photo of a city skyline", "a close-up of a flower"
    ]
    
    # Process the image and the text descriptions
    inputs = clip_processor(text=text_descriptions, images=image, return_tensors="pt", padding=True)
    
    # Get image and text features
    outputs = clip_model(**inputs)
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
    
    # Calculate similarity between image and text features
    similarity = torch.matmul(image_features, text_features.T)
    similarity_scores = similarity.squeeze(0).cpu().detach().numpy()
    
    # Get the index of the most similar text
    best_caption_idx = similarity_scores.argmax()
    
    return text_descriptions[best_caption_idx]

# Streamlit interface
st.title("GPT-3 and CLIP Image Captioning")
st.header("Chat with GPT-3 and generate image captions!")

# Chat interface
user_input = st.text_input("Enter your message:")
if user_input:
    chat_response = chat_with_gpt3(user_input)
    st.write(f"GPT-3 Response: {chat_response}")

# Image Upload
uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    caption = get_image_caption(image)
    st.write(f"Generated Caption: {caption}")
