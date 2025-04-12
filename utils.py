import pandas as pd

def preprocess_input(user_input_dict):
    df = pd.DataFrame([user_input_dict])
    return df
