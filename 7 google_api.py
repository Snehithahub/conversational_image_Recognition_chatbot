# Import necessary libraries
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import os
import textwrap
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure the Google Generative AI Gemini model
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Please set your GOOGLE_API_KEY in the environment variables.")
genai.configure(api_key=api_key)

# Function to generate a response from Gemini
def get_gemini_response(prompt, image=None):
    try:
        model = genai.GenerativeModel("gemini-pro-vision")
        if prompt and image:
            response = model.generate_content([prompt, image])
        elif prompt:
            response = model.generate_content(prompt)
        else:
            response = model.generate_content(image)
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"

# Streamlit app configuration
st.set_page_config(
    page_title="Gemini AI Image Recognition Chatbot",
    page_icon=":camera:",
    layout="wide"
)

# Inline CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #f8f8f8;
            font-family: Arial, sans-serif;
        }
        h1 {
            font-weight: bold;
        }
        button {
            background-color: #4c9dff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #3a7ec3;
        }
        textarea {
            border-radius: 4px;
            border: 1px solid #ff0000;
            padding: 10px;
            font-family: inherit;
            font-size: 14px;
        }
        textarea::placeholder {
            color: #ffffff;
        }
        .stTextArea div[data-baseweb="textarea"] textarea {
            background-color: #000000;
            color: #ffffff;
        }
        .stImage caption {
            font-style: italic;
            color: #666;
        }
        .header {
            text-align: center;
            background-color: #4c9dff;
            padding: 20px;
            border-radius: 10px;
        }
        .header h1 {
            color: white;
            margin: 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    "<div class='header'><h1>Gemini AI Image Recognition Chatbot</h1></div>",
    unsafe_allow_html=True
)

# Layout: Two columns for text input and image upload
col1, col2 = st.columns([2, 1])

# Text input in the first column
with col1:
    st.subheader("Input Prompt")
    user_prompt = st.text_area("Enter your prompt here:", height=200, placeholder="Type your prompt...")

# Image upload in the second column
with col2:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    uploaded_image = None
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

# Button to trigger response generation
if st.button("Tell me about the image"):
    if not user_prompt and not uploaded_image:
        st.warning("Please provide a prompt or upload an image.")
    else:
        with st.spinner("Generating response..."):
            response = get_gemini_response(user_prompt, uploaded_image)
        st.success("Response generated!")
        st.write(response)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.write("Powered by Gemini AI | Built with Streamlit")
