import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import openai
import os

# Load CLIP model from Hugging Face for image recognition
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# OpenAI GPT for chatbot functionality (Replace with your OpenAI API key)
openai.api_key = "your-openai-api-key"

# Function for generating responses with OpenAI GPT
def get_gpt_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # or "gpt-4" for more advanced responses
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function for image captioning using CLIP model
def get_image_caption(image):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    outputs = model.get_text_features(**inputs)
    logits_per_image = outputs.logits_per_image
    text_features = outputs.text_embeds

    similarity = torch.matmul(logits_per_image, text_features.T)
    values, indices = similarity.topk(1)

    caption = indices[0].item()  # Get the caption for the image
    return caption

# Streamlit app
st.set_page_config(page_title="AI Image Recognition and Chatbot")

st.title("AI Image Recognition and Chatbot")

# Chatbot input
user_input = st.text_input("Ask a question:")

# Image upload
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Get image caption
    caption = get_image_caption(image)
    st.write("Image Caption:", caption)

# Submit button for chatbot
if st.button("Ask the chatbot") and user_input:
    response = get_gpt_response(user_input)
    st.write(f"Chatbot Response: {response}")
