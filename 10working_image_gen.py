import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer

# Step 1: Image Recognition using EfficientNet (CNN)
def extract_image_features(img_path):
    """
    This function loads an image, preprocesses it, and extracts features using EfficientNet.
    """
    # Load pre-trained EfficientNet model
    model = models.efficientnet_b0(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    
    # Preprocess the image
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load the image and apply transformation
    img = Image.open(img_path)
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    # Extract features
    with torch.no_grad():
        features = model(img_tensor)
    
    return features

# Step 2: Caption Generation using BLIP
def generate_caption(img):
    """
    This function generates a caption for the image using the BLIP model.
    """
    # Load the BLIP model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Preprocess the image
    inputs = processor(images=img, return_tensors="pt")
    
    # Generate caption
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

# Step 3: Conversational Chatbot using DialoGPT
def chatbot_response(input_text):
    """
    This function generates a response from the chatbot using DialoGPT.
    """
    # Load pre-trained DialoGPT model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
    
    # Encode the user input and add end-of-sequence token
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    
    # Generate chatbot's response
    chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the generated response
    chatbot_output = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return chatbot_output

# Step 4: Full Pipeline Integration
def main_pipeline(img_path):
    """
    This function runs the full pipeline: image recognition, caption generation, and chatbot interaction.
    """
    # Load and process the image
    img = Image.open(img_path)
    
    # Step 1: Generate a caption for the image
    caption = generate_caption(img)
    print("Generated Caption:", caption)
    
    # Step 2: Use the caption to interact with the chatbot
    chat_response_text = chatbot_response(caption)
    print("Chatbot Response:", chat_response_text)

# Example: Running the full pipeline with an image
img_path = "/content/images1.jfif"  # Replace with the path to your image
main_pipeline(img_path)
