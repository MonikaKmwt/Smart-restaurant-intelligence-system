

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    return pd.read_csv("data/featured_restaurants.csv")

def cuisine_text(df):
    cuisine_cols = [
        "south_indian_or_not", "north_indian_or_not",
        "fast_food_or_not", "street_food",
        "biryani_or_not", "bakery_or_not"
    ]
    def row_to_text(row):
        cuisines = []
        for col in cuisine_cols:
            if col in df.columns and row[col] == 1:
                cuisines.append(col.replace("_or_not", "").replace("_"," "))
        if not cuisines:
            return "other"
        return " ".join(cuisines) + " " + row["location"]
    df["cuisine_text"] = df.apply(row_to_text, axis=1)
    return df

def compute_similarity_matrix(df):
    #Tf-Idf on cuisine_text
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["cuisine_text"])

    #Numeric features
    numeric_features = df[[
        "norm_rating",
        "norm_price",
        "norm_delivery_time"
    ]].values

    #Combine Both
    from scipy.sparse import hstack
    #combined_matrix = hstack([tfidf_matrix, numeric_features])
    #For boosting cuisine import (weight the features): cuisine importance is high(user preference)
    combined_matrix = hstack([tfidf_matrix*2, numeric_features])

    #Cosine Similarity
    similarity_matrix = cosine_similarity(combined_matrix)
    return similarity_matrix

def get_similar_restaurants(df, similarity_matrix, restaurant_name, top_n=5):
    #Find index
    matches = df[df["restaurant_name"].str.lower() == restaurant_name.lower()]
    if matches.empty:
        print("Restaurants not found.")
        return pd.DataFrame()
    idx = matches.index[0]

    #Get similarity scores, Sort by similarity, Exclude itself
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    indices = [i[0] for i in sim_scores]

    return df.iloc[indices][[
        "restaurant_name", "rating",
        "average_price", "average_delivery_time"
    ]]

def main():
    df = load_data()
    df = cuisine_text(df)
    similarity_matrix = compute_similarity_matrix(df)
    #print("\n-------------  Similarity Matrix is Created.    ---------------")

    print("\n---------------   Similar Restaurants   -----------------\n")
    results = get_similar_restaurants(
        df, similarity_matrix,
        restaurant_name = "#Dilliwaala6", top_n=5
    )
    print(results)

    #print(df[["restaurant_name", "cuisine_text"]].head(10))

if __name__ == "__main__":
    main()