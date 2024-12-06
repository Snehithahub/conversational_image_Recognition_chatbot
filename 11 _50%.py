import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer

# Step 1: Load BLIP for Image Captioning
def generate_caption(img_path):
    """
    This function generates a caption for the image using BLIP.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Open and preprocess the image
    img = Image.open(img_path)
    inputs = processor(images=img, return_tensors="pt")
    
    # Generate caption
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Step 2: Load DialoGPT for Conversational Responses
class Chatbot:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
        self.chat_history_ids = None

    def get_response(self, input_text):
        """
        Generates a chatbot response based on user input and conversation history.
        """
        # Encode the user input
        new_user_input_ids = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')

        # Concatenate new input with chat history
        self.chat_history_ids = (
            new_user_input_ids
            if self.chat_history_ids is None
            else torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1)
        )

        # Generate response
        chat_history_ids = self.model.generate(
            self.chat_history_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode and return response
        bot_response = self.tokenizer.decode(
            chat_history_ids[:, self.chat_history_ids.shape[-1]:][0], skip_special_tokens=True
        )
        return bot_response

# Step 3: Full Pipeline with Continuous Interaction
def main_pipeline():
    """
    Main function to manage the continuous interaction pipeline.
    """
    chatbot = Chatbot()
    print("Hello! I am your assistant. You can provide an image or ask a question.")
    
    while True:
        # Prompt user for input
        user_input = input("\nYou: ")
        
        # Exit condition
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        # Detect if input is an image path
        if user_input.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                # Step 1: Process image and generate a caption
                caption = generate_caption(user_input)
                print(f"Image Caption: {caption}")
                
                # Step 2: Start a conversation with the image caption
                response = chatbot.get_response(caption)
                print(f"Chatbot: {response}")
            except Exception as e:
                print(f"Error processing image: {e}")
        else:
            # Step 3: Continue conversation with text input
            response = chatbot.get_response(user_input)
            print(f"Chatbot: {response}")

# Run the pipeline
main_pipeline()
