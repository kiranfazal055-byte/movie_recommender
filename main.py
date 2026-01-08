import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib  # For fuzzy matching

# Load data (cache for performance)
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    
    # Merge on title
    movies = movies.merge(credits, on="title")
    
    # Select relevant features
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    
    # Fill NaN and combine text features
    movies['overview'] = movies['overview'].fillna('')
    movies['genres'] = movies['genres'].apply(lambda x: " ".join([i['name'] for i in eval(x)]))
    movies['keywords'] = movies['keywords'].apply(lambda x: " ".join([i['name'] for i in eval(x)]))
    movies['cast'] = movies['cast'].apply(lambda x: " ".join([i['name'] for i in eval(x)[:3]]))  # Top 3 cast
    movies['director'] = movies['crew'].apply(lambda x: [i['name'] for i in eval(x) if i['job'] == 'Director'][0] if any(i['job'] == 'Director' for i in eval(x)) else '')
    
    movies['combined_features'] = movies['overview'] + " " + movies['genres'] + " " + movies['keywords'] + " " + movies['cast'] + " " + movies['director']
    
    return movies

# Build similarity matrix (cache it)
@st.cache_resource
def build_similarity(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim

# Recommendation function
def get_recommendations(title, movies, cosine_sim):
    # Fuzzy match for close titles
    close_matches = difflib.get_close_matches(title, movies['title'], n=1, cutoff=0.6)
    if not close_matches:
        return None
    title = close_matches[0]
    
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10
    
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'genres', 'cast', 'director']].iloc[movie_indices]

# Main app
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Interactive Movie Recommendation System")
st.markdown("Search for a movie and get personalized recommendations based on plot, genres, cast, and director!")

movies = load_data()
cosine_sim = build_similarity(movies)

# Sidebar for input
st.sidebar.header("Find Recommendations")
user_input = st.sidebar.text_input("Enter a movie title:", "Inception")

if st.sidebar.button("Get Recommendations"):
    recommendations = get_recommendations(user_input, movies, cosine_sim)
    
    if recommendations is None:
        st.error("Movie not found! Try a similar spelling.")
    else:
        st.success(f"Recommendations for **{user_input}**")
        st.dataframe(recommendations.style.background_gradient(cmap='viridis'), use_container_width=True)
        
        # Show top movies overview
        st.subheader("Explore Top Rated Movies")
        top_movies = movies.nlargest(10, 'movie_id')[['title', 'genres', 'cast']]  # Simplified
        st.table(top_movies)

# Extra interactive elements
st.sidebar.markdown("### Dataset Info")
st.sidebar.info(f"Loaded {len(movies)} movies from TMDB dataset.")

st.markdown("---")
st.caption("Built with Streamlit • Uses TF-IDF + Cosine Similarity for content-based recommendations")