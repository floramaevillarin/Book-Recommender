import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors

# Load data
cratings = pd.read_csv('data/ratings.csv')
books = pd.read_csv('data/books_cleaned.csv')

# Set the page config to make it centered
st.set_page_config(page_title="Book Recommender", page_icon="📔", layout="centered", initial_sidebar_state="auto")

@st.cache_data
def read_book_data():
    return pd.read_csv('data/books_cleaned.csv')

@st.cache_data
def read_ratings_data():
    return pd.read_csv('data/ratings.csv')

def content(books):
    books['content'] = books[['authors', 'title', 'genres', 'description']].fillna('').agg(' '.join, axis=1)
    tf_content = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.01, stop_words='english')
    tfidf_matrix = tf_content.fit_transform(books['content'])
    cosine = linear_kernel(tfidf_matrix, tfidf_matrix)
    index = pd.Series(books.index, index=books['title'])
    return cosine, index

def improved_recommendation(books, title, n=5):
    # Check if the book title exists in the dataset
    if title not in books['title'].values:
        st.error(f"The book '{title}' is not found in the dataset.")
        return pd.DataFrame()  # Return an empty DataFrame if the title is not found

    cosine_sim, indices = content(books)
    idx = indices[title]

    # Ensure the index is valid and within bounds
    if idx >= len(cosine_sim):
        st.error(f"Index {idx} is out of bounds for the cosine similarity matrix.")
        return pd.DataFrame()  # Return an empty DataFrame if the index is invalid

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    book_indices = [i[0] for i in sim_scores]

    books2 = books.iloc[book_indices][['book_id', 'title', 'authors', 'average_rating', 'ratings_count', 'genres', 'pages']]

    # Add similarity score to the recommendations
    books2['content_similarity_score'] = [score for _, score in sim_scores]
    books2 = books2.sort_values('content_similarity_score', ascending=False)

    return books2[['title', 'authors', 'average_rating', 'content_similarity_score', 'pages', 'genres']]


def collaborative_filtering_knn(books, title, n=5):
    book_id = books[books['title'] == title]['book_id'].values[0]
    users_who_read = cratings[cratings['book_id'] == book_id]['user_id'].unique()
    user_ratings = cratings[cratings['user_id'].isin(users_who_read)]
    pivot = user_ratings.pivot_table(index='user_id', columns='book_id', values='rating').fillna(0)
    
    knn = NearestNeighbors(n_neighbors=n, metric='cosine', algorithm='brute')
    knn.fit(pivot.T)
    
    book_index = pivot.columns.get_loc(book_id)
    distances, indices = knn.kneighbors(pivot.T.iloc[book_index].values.reshape(1, -1), n_neighbors=n+1)
    
    recommended_books = pivot.columns[indices.flatten()[1:]]
    
    # Ensure we don't exceed the number of available books
    if len(recommended_books) < n:
        recommended_books = list(recommended_books) * (n // len(recommended_books)) + list(recommended_books)[:(n % len(recommended_books))]
    
    recommended_books_info = books[books['book_id'].isin(recommended_books[:n])][['title', 'authors', 'average_rating', 'pages', 'genres']]
    
    # Add similarity score (from KNN distances)
    # Ensure we take the correct number of distances to match the recommendations
    num_recs = min(len(recommended_books_info), len(distances.flatten()) - 1)  # Exclude the book itself
    recommended_books_info['knn_similarity_score'] = 1 - distances.flatten()[1:num_recs+1]  # Higher distance means lower similarity
    
    # If fewer than 'n' recommendations, pad with NaN values for similarity
    if len(recommended_books_info) < n:
        additional_rows = pd.DataFrame({
            'title': [''] * (n - len(recommended_books_info)),
            'authors': [''] * (n - len(recommended_books_info)),
            'average_rating': [None] * (n - len(recommended_books_info)),
            'knn_similarity_score': [None] * (n - len(recommended_books_info)),
            'pages': [None] * (n - len(recommended_books_info)),
            'genres': [None] * (n - len(recommended_books_info))
        })
        
        # Use pd.concat() instead of append()
        recommended_books_info = pd.concat([recommended_books_info, additional_rows], ignore_index=True)
    
    recommended_books_info = recommended_books_info.sort_values(by='knn_similarity_score', ascending=False)
    
    return recommended_books_info[['title', 'authors', 'average_rating', 'knn_similarity_score', 'pages', 'genres']]



def combined_recommendations(books, title, n=5, content_weight=0.7, knn_weight=0.3):
    # Get content-based recommendations
    content_recs = improved_recommendation(books, title, n)
    content_recs['weight'] = content_recs['content_similarity_score'] * content_weight
    
    # Ensure 'genres' and 'pages' are in content_recs
    content_recs = content_recs[['title', 'authors', 'average_rating', 'content_similarity_score', 'weight', 'genres', 'pages']]
    
    # Get collaborative filtering (KNN) recommendations
    knn_recs = collaborative_filtering_knn(books, title, n)
    knn_recs['weight'] = knn_recs['knn_similarity_score'] * knn_weight
    
    # Ensure 'genres' and 'pages' are in knn_recs
    knn_recs = knn_recs[['title', 'authors', 'average_rating', 'knn_similarity_score', 'weight', 'genres', 'pages']]
    
    # Merge the two recommendation lists, weighted by their similarity scores
    combined_recs = pd.concat([content_recs, knn_recs])
    
    # Weigh the similarity scores and sort
    combined_recs['final_score'] = combined_recs['weight']
    combined_recs = combined_recs.sort_values(by='final_score', ascending=False)
    
    # Limit to the number of books selected
    combined_recs = combined_recs.head(n)  # This ensures only 'n' books are shown
    
    # Drop duplicate titles, keeping the best recommendation for each title
    return combined_recs[['title', 'authors', 'average_rating', 'final_score', 'weight', 'genres', 'pages']].drop_duplicates(subset='title')

def simple_recommender(books, n=5):
    v = books['ratings_count']
    m = books['ratings_count'].quantile(0.95)
    R = books['average_rating']
    C = books['average_rating'].median()
    score = (v / (v + m) * R) + (m / (m + v) * C)
    books['final_score'] = score
    qualified = books.sort_values('final_score', ascending=False)
    return qualified[['title', 'authors', 'average_rating', 'final_score', 'pages', 'genres']].head(n)

def main():
    # Header contents
    st.write('# Book Recommender')

    # Load book and ratings data
    books = read_book_data().copy()
    user_rating = read_ratings_data().copy()

    # Number of books to recommend
    book_num = st.selectbox('Number of books', [5, 10, 15, 20])

    # Genre filter (dropdown)
    selected_genre = st.selectbox('Select genre', ['All'] + sorted(set(genre.strip().capitalize() for genre in books['genres'].str.split(',').explode().unique())))

    # Page filter (range: below 200, 200-400, 400+)
    page_category = st.radio('Select page category', ['All', 'Below 200', '200-400', '400+'])

    # Convert the 'pages' column to numeric values (important for filtering)
    books['pages'] = pd.to_numeric(books['pages'], errors='coerce')

    # Drop rows where pages is NaN (important to clean up invalid data)
    books = books.dropna(subset=['pages'])

    # Apply genre filter if any
    if selected_genre != 'All':
        books = books[books['genres'].str.contains(selected_genre, case=False, na=False)]

    # Apply page filter
    if page_category == 'Below 200':
        books = books[books['pages'] < 200]
    elif page_category == '200-400':
        books = books[(books['pages'] >= 200) & (books['pages'] <= 400)]
    elif page_category == '400+':
        books = books[books['pages'] > 400]

    # Select book for recommendations, allow for blank (empty) selection
    selected_book = st.selectbox('Pick a book to use as your reference', [''] + list(books['title'].unique()))  # Adding an empty option
    
    # Add a "Recommend" button
    if st.button('Recommend'):
        if selected_book:
            # If a book is selected, use combined recommendations
            recommendations = combined_recommendations(books, selected_book, n=book_num)
            st.write(f'### Recommendations:')
            st.write(recommendations)
        else:
            # If no book is selected, use simple recommender
            st.write('### Recommendations:')
            simple_recs = simple_recommender(books, n=book_num)
            st.write(simple_recs)
    
if __name__ == "__main__":
    main()
