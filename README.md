# 🍽️ AI-Powered Restaurant Intelligence System

An intelligent restaurant recommendation system built using Machine Learning and deployed with Streamlit.

---

## 🚀 Project Overview

This project recommends restaurants based on user preferences such as:

* Cuisine
* Budget
* Rating
* Delivery time

It uses a **hybrid recommendation system** combining:

* Smart scoring logic
* Content-based filtering
* Similarity modeling (TF-IDF + Cosine Similarity)

---

## 🧠 Key Features

### 🔥 Smart Scoring System

Restaurants are ranked using a weighted formula:

Final Score =
(Rating × 0.5) +
(Fast Delivery × 0.3) +
(Affordable Price × 0.2)

---

### 🍽️ Hybrid Recommendation Engine

* Filters by city, price, rating
* Cuisine-based boosting (not hard filtering)
* Dynamic ranking

---

### 🤖 Similar Restaurants (ML Feature)

* TF-IDF on cuisine text
* Cosine similarity
* Hybrid similarity using numeric features

---

### 🖥️ Interactive Streamlit App

* Sidebar filters
* Apply button (controlled execution)
* Clean UI cards
* “Show Similar” feature inline

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit

---

## 📊 Dataset

* 27,000+ restaurants
* Features:

  * Rating
  * Price
  * Delivery time
  * Cuisine categories
  * Location

---

## ⚙️ How to Run Locally

```bash
git clone <your-repo-link>
cd smart-restaurant-intelligence
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌍 Live Demo

👉 (Add your Streamlit link here after deployment)

---

## 🎯 Future Improvements

* Add dashboard (EDA visualizations)
* Improve UI (cards grid layout)
* Use advanced embeddings (BERT)
* Add user login + personalization

---

## 👨‍💻 Author

Monika kumawat
