import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def load_data():
    df = pd.read_csv("data/OnlineNewsPopularity.csv")
    df = df.drop(['url', 'timedelta'], axis=1)
    return df
def train_model():
    df = load_data()
    X = df.drop("shares", axis=1)
    y = df["shares"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    return model, mae

