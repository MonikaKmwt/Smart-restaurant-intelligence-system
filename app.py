#UI

import streamlit as st
import pandas as pd
from src.recommender import get_recommendations
from src.similarity_model import (
    load_data, cuisine_text,
    compute_similarity_matrix, get_similar_restaurants
)

#----Page Configuration
st.set_page_config(
    page_title="Smart Restaurants Intelligence System", layout="wide"
)

#----Cache
@st.cache_data
def load_cached_data():
    df = load_data()
    return cuisine_text(df)

@st.cache_resource
def load_similarity(df):
    return compute_similarity_matrix(df)

#----Load Data
df = load_cached_data()
similarity_matrix = load_similarity(df)

#----Session State'
if "selected_restaurant" not in st.session_state:
    st.session_state.selected_restaurant = None

if "results" not in st.session_state:
    st.session_state.results = None

#----Title
st.title(" 🍽️ Smart Restaurants Intelligence System")
st.markdown("Find the best restaurants 🍽️ For You 😄")

#----Sidebar Form
st.sidebar.header("Filters")
with st.sidebar.form("filter_form"):
    cities = sorted(df["location"].unique())
    selected_city = st.selectbox("Select city", cities)
    cuisine_options = [
    "south_indian", "north_indian",
    "fast_food", "street_food",
    "biryani", "bakery"
    ]
    selected_cuisines = st.multiselect("Select cuisines", cuisine_options)
    max_price = st.slider("Max price", 50, 1000, 300)
    min_rating = st.slider("Min rating", 1.0, 5.0, 3.5)
    #----Apply Button
    submit = st.form_submit_button("Apply Filters")

#----Handle submit
if submit:
    #----RESET previous selection
    st.session_state.selected_restaurant = None
    #----Store results
    st.session_state.results = get_recommendations(
        city=selected_city,
        cuisines=selected_cuisines,
        max_price=max_price,
        min_rating=min_rating,
        top_n=10
    )

#----Placeholder
if st.session_state.results is None:
    st.info("Please select filters (city, cuisines, etc.) and click **Apply Filters** to see recommendations.")

#----Recommendations
if st.session_state.results is not None:
    results = st.session_state.results

    if results.empty:
        st.warning("No restaurants found with selected filters.")
    else:
        st.subheader("Top Recommendations")

        for _, row in results.iterrows():
            with st.container():
                st.markdown(f"""
                ### 🍽️{row['restaurant_name']}  
                ⭐ Rating: **{row['rating']}**   
                💸 Average Price: ₹{row['average_price']}  
                🚚 Average Delivery Time: {row['average_delivery_time']} mins  
                📊 Score: **{round(row['final_score'], 3)}**  
                """)
                #----Similar restaurants button
                if st.button(f"Show Similar for {row['restaurant_name']}", key=row['restaurant_name']):
                    st.session_state.selected_restaurant = row['restaurant_name']
                st.markdown("---")

                #----Similar Restaurants
                if st.session_state.selected_restaurant == row['restaurant_name']:
                    st.markdown("## Similar Restaurants")
                    similar = get_similar_restaurants(
                        df, similarity_matrix,
                        restaurant_name = st.session_state.selected_restaurant,
                        top_n = 5
                    )
                    for _, sim_row in similar.iterrows():
                        st.markdown(f"""
                        <div style="
                            padding: 12px;
                            border-radius: 10px;
                            background-color: #1e1e1e;
                            margin-bottom: 10px;">
                        <b>🍽️ {sim_row['restaurant_name']}</b><br>
                        ⭐ {sim_row['rating']} |
                        💸 {sim_row['average_price']} |
                        🚚 {sim_row['average_delivery_time']} mins |
                        </div>
                        """, unsafe_allow_html=True)