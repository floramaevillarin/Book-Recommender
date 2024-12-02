import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

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
    cosine_sim, indices = content(books)
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]  # Get the top n recommendations
    book_indices = [i[0] for i in sim_scores]
    
    books2 = books.iloc[book_indices][['book_id', 'title', 'authors', 'average_rating', 'ratings_count', 'genres', 'pages', 'year']]
    
    # Add similarity score to the recommendations
    books2['content_similarity_score'] = [score for _, score in sim_scores]
    books2 = books2.sort_values('content_similarity_score', ascending=False)
    
    return books2[['title', 'authors', 'average_rating', 'content_similarity_score', 'pages', 'genres', 'year']]

def collaborative_filtering_svd(books, title, n=5):
    book_id = books[books['title'] == title]['book_id'].values[0]
    
    # Create a user-item matrix
    user_ratings = cratings.pivot_table(index='user_id', columns='book_id', values='rating').fillna(0)
    
    # Perform SVD
    svd = TruncatedSVD(n_components=50, random_state=42)  # Adjust n_components based on your dataset size
    svd_matrix = svd.fit_transform(user_ratings)
    
    # Compute cosine similarity
    book_sim = cosine_similarity(svd_matrix.T)  # Transpose to align books in rows
    
    # Find the index of the selected book
    book_index = list(user_ratings.columns).index(book_id)
    
    # Get similarity scores for the selected book
    sim_scores = list(enumerate(book_sim[book_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    
    # Get the book IDs for the top recommendations
    recommended_books = [user_ratings.columns[i[0]] for i in sim_scores]
    
    # Fetch book details
    recommendations = books[books['book_id'].isin(recommended_books)][['title', 'authors', 'average_rating', 'pages', 'genres', 'year']]
    
    # Add similarity scores to recommendations
    recommendations['svd_similarity_score'] = [score for _, score in sim_scores]
    recommendations = recommendations.sort_values('svd_similarity_score', ascending=False)
    
    return recommendations[['title', 'authors', 'average_rating', 'svd_similarity_score', 'pages', 'genres', 'year']]

def combined_recommendations(books, title, n=5, content_weight=0.7, svd_weight=0.3):
    # Get content-based recommendations
    content_recs = improved_recommendation(books, title, n)
    content_recs['weight'] = content_recs['content_similarity_score'] * content_weight
    
    # Get collaborative filtering (SVD) recommendations
    svd_recs = collaborative_filtering_svd(books, title, n)
    svd_recs['weight'] = svd_recs['svd_similarity_score'] * svd_weight
    
    # Merge and weigh recommendations
    combined_recs = pd.concat([content_recs, svd_recs])
    combined_recs['final_score'] = combined_recs['weight']
    combined_recs = combined_recs.sort_values(by='final_score', ascending=False)
    
    return combined_recs[['title', 'authors', 'average_rating', 'final_score', 'weight', 'genres', 'pages']].drop_duplicates(subset='title')



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

    # Select book for recommendations
    selected_book = st.selectbox('Select a book', books['title'].unique())
    
    # Add a "Recommend" button
    if st.button('Recommend'):
        if selected_book:
            recommendations = combined_recommendations(books, selected_book, n=book_num)
            st.write(f'### Recommendations based on {selected_book}')
            st.write(recommendations)
        else:
            st.write('Please select a book to get recommendations.')
    
if __name__ == "__main__":
    main()
