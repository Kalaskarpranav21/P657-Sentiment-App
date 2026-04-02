import streamlit as st
import joblib
import re

# --- 1. Load the Saved Model and Vectorizer ---
# We use joblib since that's what was used to save them in your notebook
@st.cache_resource # This prevents reloading the model every time the user types
def load_components():
    model = joblib.load('sentiment_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf

model, tfidf = load_components()

# --- 2. Define the Text Cleaning Function ---
# CRITICAL: This must match exactly how you cleaned the text in your Jupyter Notebook!
def clean_text(text):
    text = str(text).lower() # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
    # Add your stopword removal here if you used it during training!
    return text

# --- 3. Build the Streamlit User Interface ---
# Page configuration
st.set_page_config(page_title="Product Sentiment Analyzer", page_icon="🛍️")

st.title("🛍️ E-Commerce Product Sentiment Analyzer")
st.write("Enter a customer review below to instantly predict its sentiment.")

# Create a text area for user input
user_review = st.text_area("Type the product review here:", height=150)

# Create a button to trigger the prediction
if st.button("Analyze Sentiment", type="primary"):
    
    if user_review.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing..."):
            # Step A: Clean the user's input
            cleaned_input = clean_text(user_review)
            
            # Step B: Vectorize the text using your loaded TF-IDF
            vectorized_input = tfidf.transform([cleaned_input])
            
            # Step C: Make the prediction
            prediction = model.predict(vectorized_input)[0]
            
            # Step D: Display the result beautifully
            st.markdown("### Result:")
            if prediction == 'Positive':
                st.success(f"✅ **{prediction} Sentiment** - The customer seems happy!")
            elif prediction == 'Negative':
                st.error(f"❌ **{prediction} Sentiment** - The customer is unsatisfied.")
            else:
                st.info(f"😐 **{prediction} Sentiment** - The review is neutral.")