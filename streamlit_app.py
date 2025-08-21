# import packages
from dotenv import load_dotenv
import openai
import streamlit as st
import os

# load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@st.cache
def get_response(user_prompt, temperature):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use the latest chat model
            messages=[
                {"role": "user", "content": user_prompt}  # Prompt
            ],
            temperature=temperature,  # A bit of creativity
            max_tokens=100  # Limit response length
        )
        return response
    except Exception as e:
        return f"Error: {e}"

st.title("Hello, GenAI!")
st.write("This is your first Streamlit app.")

# Add a text input box for the user prompt
user_prompt = st.text_input("Enter your prompt:", "Explain generative AI in one sentence.")

# Add a slider for temperature
temperature = st.slider(
    "Model temperature:",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01,
    help="Controls randomness: 0 = deterministic, 1 = very creative"
)

# Add a button to trigger the API call
if st.button("Get AI Response"):
    with st.spinner("AI is working..."):
        response = get_response(user_prompt, temperature)
        if isinstance(response, str) and response.startswith("Error:"):
            st.error(response)
        else:
            st.write("**AI Response:**")
            st.write(response.choices[0].message.content)
