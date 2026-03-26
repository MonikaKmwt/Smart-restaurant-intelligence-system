# GOAL AND UNDERSTANDING THE DATA

import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    return df

def explore_data(df):
    print("Data Shape:", df.shape)
    print("\n Columns:", df.columns)
    print("Data Info:", df.info)
    print("Missing Values:\n", df.isnull().sum())
    print("First 5 Rows:\n", df.head(3))

def clean_column_names(df):
    df.columns = (
        df.columns
        .str.strip()                          # remove outer spaces
        .str.lower()
        .str.replace(r"\s+", "_", regex=True) # replace ANY spaces with _
        .str.replace(r"_+", "_", regex=True)  # fix multiple underscores
    )
    return df

def clean_data(df):
    # Example cleaning steps - modify as needed
    print(df.shape)
    df = df.dropna()  # Remove rows with missing values
    print(df.shape)
    print("There are no missing values in the dataset.")
    df = df.drop_duplicates()  # Remove duplicate rows
    return df

def main():
    input_path = "data/2indian_restaurants.csv"
    df = load_data(input_path)
    df = clean_column_names(df)
    explore_data(df)
    df = clean_data(df)
    print("After preprocessing:", df.shape)
    print(df.head(10))
    #save cleaned data
    df.to_csv("data/cleaned_restaurants.csv", index=False)

if __name__ == "__main__":
    main()  
