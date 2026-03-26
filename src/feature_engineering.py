#FEATURE ENGINEERING &
#SMART SCORING

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data():
    return pd.read_csv("data/cleaned_restaurants.csv")

def normalize_features(df):
    scaler = MinMaxScaler()

    #Normalize rating (higher is better)
    df['norm_rating'] = scaler.fit_transform(df[['rating']])

    #Normalize price (lower is better, so we take the inverse)
    df['norm_price'] = scaler.fit_transform(df[['average_price']])

    #Normalize delivery time (lower is better, so we take the inverse)
    df['norm_delivery_time'] = scaler.fit_transform(df[['average_delivery_time']])

    #Inverse price and delivery time to reflect that lower values are better
    df['norm_price'] = 1 - df['norm_price']
    df['norm_delivery_time'] = 1 - df['norm_delivery_time']

    return df

def compute_scores(df, weights=None):
    if weights is None:
        weights = {
            "rating": 0.5,
            "delivery": 0.3,
            "price":0.2}
    df['Score'] = (
        df['norm_rating']*weights['rating'] +
        df["norm_delivery_time"] * weights["delivery"] +
        df["norm_price"] * weights["price"]
    )
    return df

def main():
    df = load_data()
    df = normalize_features(df)
    df = compute_scores(df)

    print("Sample with Scores:\n")
    print(df[[
        "restaurant_name", "rating", "average_price",
        "average_delivery_time", "Score"
    ]].head())

    df.to_csv("data/featured_restaurants.csv", index=False)

if __name__ == "__main__":
    main()

