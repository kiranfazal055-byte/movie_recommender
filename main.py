import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    # Extract year from title and clean title
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    movies['title_clean'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
    # Combine genres (already pipe-separated)
    movies['genres'] = movies['genres'].str.replace('|', ' ')
    movies['combined_features'] = movies['genres'] + " " + movies['title_clean']
    return movies

# Build similarity
@st.cache_resource
def build_similarity(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim

# Get recommendations
def get_recommendations(title, movies, cosine_sim):
    # Fuzzy match
    close_matches = difflib.get_close_matches(title, movies['title'], n=1, cutoff=0.5)
    if not close_matches:
        return None
    title = close_matches[0]
    
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10
    
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'genres', 'year']].iloc[movie_indices]

# App
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")
st.markdown("Enter a movie title (e.g., Toy Story, Inception) to get similar recommendations based on genres!")

movies = load_data()
cosine_sim = build_similarity(movies)

st.sidebar.header("Search")
user_input = st.sidebar.text_input("Movie title:", "Toy Story")

if st.sidebar.button("Recommend"):
    recommendations = get_recommendations(user_input, movies, cosine_sim)
    
    if recommendations is None:
        st.error("Movie not found! Check spelling or try a similar title.")
    else:
        st.success(f"Recommendations for **{user_input}**")
        st.dataframe(recommendations.style.background_gradient(cmap='viridis'), use_container_width=True)

st.sidebar.info(f"Dataset: MovieLens (9,700+ movies)")
st.caption("Built with Streamlit • Content-based filtering using TF-IDF + Cosine Similarity")
