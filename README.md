# 📚 Book Recommendation System  

## 📝 Overview  
This project implements a **hybrid book recommendation system** that combines **content-based filtering** and **collaborative filtering** techniques. It processes book ratings and metadata to suggest books that a user might enjoy, leveraging both **machine learning models** and **data-driven insights**.  

### 🔍 Problem Statement  
With the vast number of books available today, users often struggle to find relevant books based on their preferences. A **recommendation system** helps by analyzing book features and user interactions to generate personalized suggestions.  

## 📂 Dataset  
The project utilizes the **Book Recommendation Dataset** from Kaggle, which consists of: **Books.csv** (metadata such as title, author, publication year, and publisher), **Ratings.csv** (user ratings for books on a scale of 1 to 10), and **Users.csv** (demographic information about users).  

## 🔧 Features and Methodology  

### 🧹 1. Data Cleaning & Preprocessing  
**Handling Missing Values** by removing null values from metadata, **Feature Selection** by keeping only relevant columns such as title, author, and publication year, **Data Normalization** by converting book titles and authors to lowercase and removing spaces, **Filtering Active Users** who have rated at least **100 books**, and **Filtering Popular Books** with at least **25 ratings**.  

### 📖 2. Content-Based Filtering  
**Text Preprocessing** involves tokenization using **NLTK**, stemming, and lemmatization. **Feature Engineering** combines book title, author, and publication year into a single **metadata vector** and applies **TF-IDF Vectorization** to convert text into numerical data. **Similarity Calculation** uses **cosine similarity** to compute distances between books and identify **top N similar books** to the input book.  

### 🤝 3. Collaborative Filtering  
**User-Item Interaction Matrix** creates a pivot table where rows represent books, columns represent users, and values are ratings. **User-Based Recommendations** identify users with similar reading patterns. The **K-Nearest Neighbors (KNN) Algorithm** finds books rated similarly by other users and returns books with the highest **similarity scores**.  

### 🔗 4. Hybrid Recommendation System  
**Combines Content-Based and Collaborative Filtering** by assigning adjustable weights (`content_weight` and `collaborative_weight`), normalizing similarity scores, and generating **personalized hybrid recommendations**.  

### 📊 5. Evaluation & Performance Metrics  
**Baseline Models** compare the hybrid method against **Global Mean Predictor** (predicts average rating for all books), **User Mean Predictor** (predicts using the average rating per user), and **Item Mean Predictor** (predicts using the average rating per book). **Root Mean Squared Error (RMSE)** is used to measure accuracy across different approaches.  

## 🚀 Installation & Setup  

### Prerequisites  
Ensure you have **Python 3.x** installed along with the required dependencies.  

### 🔹 Step 1: Install Dependencies  
Run the following command:  
```bash
pip install numpy pandas nltk scikit-learn kagglehub matplotlib
```
## 🔹 Step 2: Download Dataset
The dataset is automatically fetched using `kagglehub`.

## 🔹 Step 3: Run the Code
Execute the Jupyter Notebook or Python script to generate recommendations.

## 🔨 Usage

### 🟢 Content-Based Filtering
To get top 5 similar books based on metadata, use:

```python
contentFiltering(books, "1984",top_n=5)
```
### 🟠 Hybrid Recommendations
To combine both approaches for a more accurate recommendation, use:
```python
hybrid_recommendations(
    user_id=277427, 
    book_title="1984", 
    content_weight=0.5, 
    collaborative_weight=0.5, 
    num_recommendations=10
)
```

## 🧪 Evaluation
To compute RMSE scores for different recommendation models, use:

```python
compare_baseline(userItem, test_indices)
```
## 📈 Results  
**The hybrid model improves recommendation accuracy** by leveraging both content similarity and user behavior.  
**RMSE evaluation confirms** that the hybrid method outperforms individual approaches.  
**User engagement is higher** due to personalized recommendations.  

## 🛠️ Future Improvements  
- **Implement deep learning models** for better predictions.  
- **Integrate real-time user feedback** for dynamic recommendations.  
- **Expand the dataset** with new book releases and user preferences. 

