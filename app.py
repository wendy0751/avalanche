# import packages
import streamlit as st
import pandas as pd
import re
import os
import altair as alt
from dotenv import load_dotenv
import openai

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

# Helper function to get dataset path
def get_dataset_path():
    """Function to get dataset path - you can modify this as needed"""
    return "data/customer_reviews.csv"  # Update this path to your actual dataset

# Helper function to clean text
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\\w\\s]', '', text)
    return text

st.title("Hello, GenAI!")
st.write("This is your GenAI-powered data processing app.")

# Layout two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Ingest Dataset"):
        try:
            csv_path = get_dataset_path()
            st.session_state["df"] = pd.read_csv(csv_path)
            st.success("Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path.")

with col2:
    if st.button("🧹 Parse Reviews"):
        if "df" in st.session_state:
            st.session_state["df"]["CLEANED_SUMMARY"] = st.session_state["df"]["SUMMARY"].apply(clean_text)
            st.success("Reviews parsed and cleaned!")
        else:
            st.warning("Please ingest the dataset first.")

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

# Display the dataset if it exists
if "df" in st.session_state:
    # Product filter dropdown
    st.subheader("🔍 Filter by Product")
    product = st.selectbox("Choose a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))
    st.subheader(f"📁 Reviews for {product}")

    if product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]
    st.dataframe(filtered_df)
    
    st.subheader(f"Sentiment Score Distribution for {product}")
    # Create Altair histogram using add_params instead of add_selection
    interval = alt.selection_interval()
    chart = alt.Chart(filtered_df).mark_bar().add_selection(
        interval
    ).encode(
        alt.X("SENTIMENT_SCORE:Q", bin=alt.Bin(maxbins=10), title="Sentiment Score"),
        alt.Y("count():Q", title="Frequency"),
        tooltip=["count():Q"],
        color=alt.condition(interval, alt.value('steelblue'), alt.value('lightgray'))
    ).properties(
        width=600,
        height=400,
        title="Distribution of Sentiment Scores"
    )
    st.altair_chart(chart, use_container_width=True)
