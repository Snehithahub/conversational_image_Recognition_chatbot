import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, CLIPProcessor, CLIPModel
import torch
from PIL import Image

# Load GPT-2 model and tokenizer
gpt2_model_name = "gpt2"  # You can use 'gpt2-medium', 'gpt2-large', or 'gpt2-xl' for larger models
model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

# Load CLIP model and processor for image captioning
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Function to generate text using GPT-2
def chat_with_gpt2(user_input):
    inputs = tokenizer.encode(user_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to generate image caption using CLIP
def get_image_caption(image):
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    outputs = clip_model.get_text_features(**inputs)
    return outputs

# Streamlit interface
st.title("GPT-2 and CLIP Image Captioning")
st.header("Chat with GPT-2 and generate image captions!")

# Chat interface
user_input = st.text_input("Enter your message:")
if user_input:
    chat_response = chat_with_gpt2(user_input)
    st.write(f"GPT-2 Response: {chat_response}")

# Image Upload
uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    caption_features = get_image_caption(image)
    st.write(f"Image Caption Features: {caption_features}")
