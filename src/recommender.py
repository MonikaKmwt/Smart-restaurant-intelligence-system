#RECOMMENDATION ENGINE

import pandas as pd
import numpy as np

def load_data():
    return pd.read_csv("data/featured_restaurants.csv")

'''def filter_by_cuisines(df, cuisines):
    if not cuisines:
        return df
    cuisines_cols = [f"{c}_or_not" for c in cuisines]
    #keep rows where ANY cuisine is 1
    mask = df[cuisines_cols].any(axis=1)
    return df[mask]'''

def add_cuisine_boost(df, cuisines, boost_value=0.1):
    if not cuisines:
        df["cuisine_boost"] = 0
        return df

    cuisine_cols = [f"{c}_or_not" for c in cuisines if f"{c}_or_not" in df.columns]

    if not cuisine_cols:
        df["cuisine_boost"] = 0
        return df

    # If ANY selected cuisine matches → boost
    df["cuisine_boost"] = df[cuisine_cols].any(axis=1).astype(int) * boost_value

    return df

def get_recommendations(
    city, cuisines, max_price, min_rating, top_n=10
):
    df = load_data()
    #Basic Filters
    df = df[df["location"].str.lower() == city.lower()]
    df = df[df["average_price"] <= max_price]
    df = df[df["rating"] >= min_rating]

    #Cuisine Filters
    df = add_cuisine_boost(df, cuisines)

    #new score
    df['final_score'] = df['Score'] + df['cuisine_boost']

    #sort by score
    df = df.sort_values(by="final_score", ascending=False)

    return df.head(top_n)

def main():
    results = get_recommendations(
        city="Bangalore", cuisines=['fast_food'],
        max_price=300, min_rating=3.5, top_n=5
        )

    print("------ TOP RECOMMENDATIONS -------")
    print(results[[
        "restaurant_name", "rating", "average_price",
        "average_delivery_time", "Score", "cuisine_boost", "final_score"
    ]])

if __name__ == "__main__":
    main()