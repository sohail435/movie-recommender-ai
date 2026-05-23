import streamlit as st
import pickle
import pandas as pd

# 1. Page Configuration Set Up
st.set_page_config(page_title="Movie Recommender AI", page_icon="🎬", layout="centered")

# 2. Secure Data Loading Infrastructure
@st.cache_data
def load_app_artifacts():
    """
    Safely unpacks binary data assets from disk storage into active runtime memory.
    @st.cache_data ensures the system only reads the heavy files once on startup, 
    making subsequent user searches run instantly.
    """
    # Load the dictionary mapping structure and convert it back to a Pandas DataFrame
    movies_dict = pickle.load(open('models/movie_dict.pkl', 'rb'))
    movies_df = pd.DataFrame(movies_dict)
    
    # Load the dense calculated spatial similarity angles matrix array
    similarity_matrix = pickle.load(open('models/similarity.pkl', 'rb'))
    
    return movies_df, similarity_matrix

# Execute the loader block
try:
    movies, similarity = load_app_artifacts()
except FileNotFoundError:
    st.error("Model artifacts missing! Please verify your train.py has successfully generated files inside 'models/'.")
    st.stop()

# 3. Recommendation Retrieval Algorithm Logic Block
def recommend_movies(movie_title):
    """
    Finds the index of the requested film, pulls its similarity vector, 
    sorts the absolute closest geometric matches, and returns titles.
    """
    # Dictionary Lookup: Find row positional index integer
    movie_idx = movies[movies['title'] == movie_title].index[0]
    
    # Extract row array scores and convert them into an enumerated index list of tuples
    distances = similarity[movie_idx]
    movies_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]
    
    # Accumulate the top titles
    recommended_titles = []
    for item in movies_list:
        recommended_titles.append(movies.iloc[item[0]].title)
        
    return recommended_titles

# 4. Streamlit UI Layout Presentation View
st.title("🎬 Movie Recommender AI System")
st.write("Select a movie below, and our machine learning engine will analyze thousands of data tags to suggest the closest matches!")

# Dropdown element populated with our dictionary-backed movie titles array
selected_movie = st.selectbox(
    "Type or select a movie from the catalog:",
    movies['title'].values
)

# Interactivity Activation Button
if st.button("Generate Recommendations", type="primary"):
    with st.spinner("Analyzing semantic tag similarities..."):
        recommendations = recommend_movies(selected_movie)
        
    st.success("Here are your top 5 targeted recommendations:")
    
    # Render the recommendations neatly inside numbered visual design cards
    for idx, movie in enumerate(recommendations, start=1):
        st.markdown(f"#### **{idx}. {movie}**")
