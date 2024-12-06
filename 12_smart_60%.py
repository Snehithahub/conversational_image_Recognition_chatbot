import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from PIL import Image


class SmartChatbot:
    def __init__(self):
        # Load DialoGPT for conversation
        self.chat_tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.chat_model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")

        # Load BLIP for image captioning
        self.image_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.image_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        # Memory for conversation context
        self.memory = {"image_caption": None, "user_context": ""}

    def generate_caption(self, img_path):
        """
        Generate a caption for the input image.
        """
        try:
            img = Image.open(img_path)
            inputs = self.image_processor(images=img, return_tensors="pt")
            output = self.image_model.generate(**inputs)
            caption = self.image_processor.decode(output[0], skip_special_tokens=True)

            # Save the caption to memory
            self.memory["image_caption"] = caption
            return caption
        except Exception as e:
            return f"Error processing image: {e}"

    def chat_response(self, user_input):
        """
        Generate a conversational response using DialoGPT.
        """
        # Incorporate memory for context-aware conversation
        if self.memory["image_caption"]:
            user_input = f"{self.memory['image_caption']} {user_input}"

        # Encode user input
        new_user_input_ids = self.chat_tokenizer.encode(user_input + self.chat_tokenizer.eos_token, return_tensors="pt")

        # Generate response
        bot_output_ids = self.chat_model.generate(
            new_user_input_ids,
            max_length=500,
            pad_token_id=self.chat_tokenizer.eos_token_id,
        )

        # Decode response
        bot_response = self.chat_tokenizer.decode(bot_output_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return bot_response

    def process_user_input(self, user_input):
        """
        Main handler for user input, handling text and image-based queries.
        """
        # Check if the input is an image file
        if user_input.lower().endswith((".jpg", ".jpeg", ".png")):
            caption = self.generate_caption(user_input)
            return f"Image Caption: {caption}"

        # Process text input
        elif "color" in user_input.lower() and self.memory["image_caption"]:
            return f"The object appears to be {self.memory['image_caption'].split()[-1]}."

        elif "name" in user_input.lower() and self.memory["image_caption"]:
            return "It doesn't have a name yet. You could name it!"

        else:
            return self.chat_response(user_input)

    def run(self):
        """
        Run the chatbot for continuous interaction.
        """
        print("Hello! I am your assistant. You can provide an image or ask a question.")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            response = self.process_user_input(user_input)
            print(f"Chatbot: {response}")


# Run the enhanced chatbot
chatbot = SmartChatbot()
chatbot.run()
