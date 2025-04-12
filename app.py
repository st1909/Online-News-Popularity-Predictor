import streamlit as st
from model import train_model
from utils import preprocess_input

st.title("📰 Online News Popularity Predictor")

st.markdown("Enter the article metrics to predict the number of shares:")
inputs = {
    "n_tokens_title": st.slider("Title Tokens", 5, 20, 10),
    "n_tokens_content": st.slider("Content Tokens", 100, 3000, 500),
    "num_hrefs": st.slider("Hyperlinks", 0, 50, 4),
    "num_imgs": st.slider("Images", 0, 20, 1),
    "num_videos": st.slider("Videos", 0, 10, 0),
    "average_token_length": st.slider("Average Token Length", 3, 7, 4),
    "num_keywords": st.slider("Keywords", 1, 10, 5),
    "self_reference_min_shares": st.slider("Self Ref Min Shares", 0, 2000, 300)
}
if st.button("Predict"):
    model, mae = train_model()
    input_df = preprocess_input(inputs)
    prediction = model.predict(input_df)[0]

    st.subheader(f"🔮 Predicted Shares: {int(prediction)}")
    st.caption(f"Model MAE: {int(mae)}")

