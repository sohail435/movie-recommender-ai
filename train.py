import pandas as pd
import numpy as np
import ast
import pickle

def load_and_merge_data(movies_path,credits_path):
    """
    Loads raw CSV files and merges them into a single coherent DataFram.
    """
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    df = movies.merge(credits,on='title')

    df = df[['movie_id', 'title', 'overview', 'genres','keywords', 'cast','crew']]
    df.dropna(inplace=True)
    return df

def convert_json_to_list(text_column):
    """
    Parses stringified JSON arrays to extract just the string descriptor names.
    Transforms: '[{"id":28, "name":"Action"}]'->['Action']
    """
    list_accumulator = []
    for item in ast.literal_eval(text_column):
        list_accumulator.append(item['name'])
    return list_accumulator

def get_top_3_cast(text_column):
    """
    Traverses the cast list to isolate the top three primary billing actors.
    """
    counter = 0
    actors = []

    for item in ast.literal_eval(text_column):
        if counter < 3:
            actors.append(item['name'])
            counter += 1
        else:
            break
    return actors

def get_director(text_column):
    """
    Scans the crew dictionary list to locate the single production Director.
    """
    director = []

    for item in ast.literal_eval(text_column):
        if item['job'] == 'Director':
            director.append(item['name'])
            break
    return director

def remove_spaces(word_list):
    """
    Collapses multi_word strings to maintain entity identification during vector tokenization.
    Transforms:['Sam Worthington','Sci-Fi,]
    """
    return[word.replace(" ","") for word in word_list]

def preprocess_pipeline(movies_csv,credits_csv):
    df = load_and_merge_data(movies_csv,credits_csv)

    df['genres'] = df['genres'].apply(convert_json_to_list)
    df['keywords'] = df['keywords'].apply(convert_json_to_list)
    df['cast'] = df['cast'].apply(get_top_3_cast)
    df['crew'] = df['crew'].apply(get_director)
    df['overview'] = df['overview'].apply(lambda x:x.split() if isinstance(x, str) else [])
    df['genres'] = df['genres'].apply(remove_spaces)
    df['keywords'] = df['keywords'].apply(remove_spaces)
    df['cast'] = df['cast'].apply(remove_spaces)
    df['crew'] = df['crew'].apply(remove_spaces)
    df['tags'] = df['overview']+ df['genres']+df['keywords']+df['cast']+df['crew']
    df['tags'] = df['tags'].apply(lambda x: " ".join(x).lower())
    final_cleaned_df = df[['movie_id','title','tags']]
    return final_cleaned_df

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def train_recommendation_model(cleaned_df):
    """
    Transforms clean text tags into mathematical vectors and computes 
    the spatial similarity matrix across all loaded items.
    """
    print("Initializing vectorizer matrix engineering...")
    
    # 1. Initialize the CountVectorizer engine
    # max_features=5000: Selects only the top 5000 most popular words across all movie tags
    # stop_words='english': Automatically drops filler words like 'and', 'the', 'is'
    cv = CountVectorizer(max_features=5000, stop_words='english')
    
    # 2. Transform the text paragraph column into a mathematical feature matrix array
    vector_matrix = cv.fit_transform(cleaned_df['tags']).toarray()
    
    print(f"Vector Space Matrix built successfully: {vector_matrix.shape}")
    
    # 3. Calculate the Cosine Similarity percentages across all vectors
    print("Calculating Cosine Similarity angles...")
    similarity_matrix = cosine_similarity(vector_matrix)
    
    return similarity_matrix

def export_model_artifacts(cleaned_df, similarity_matrix):
    """
    Serializes and compresses our custom data structures into binary pickle (.pkl) 
    files so our Streamlit frontend web app can access them instantly.
    """
    print("Exporting model structures to disk storage...")
    
    # We export the clean data as a native Python dictionary structure 
    # for O(1) quick hash table lookups later in the UI frontend
    pickle.dump(cleaned_df.to_dict(), open('models/movie_dict.pkl', 'wb'))
    
    # Export the calculated similarity matrix array
    pickle.dump(similarity_matrix, open('models/similarity.pkl', 'wb'))
    
    print("All model artifacts saved inside 'models/' directory successfully!")

if __name__ == "__main__":
    # Define file path dependencies
    MOVIES_DATA = "data/tmdb_5000_movies.csv"
    CREDITS_DATA = "data/tmdb_5000_credits.csv"
    
    print("--- Starting Movie Recommender AI Training Pipeline ---")
    
    # Phase 1: Text Preprocessing & Cleaning
    cleaned_data = preprocess_pipeline(MOVIES_DATA, CREDITS_DATA)
    
    # Phase 2: Vector Math Engine Execution
    similarity_scores = train_recommendation_model(cleaned_data)
    
    # Phase 3: Binary Serialization Export
    export_model_artifacts(cleaned_data, similarity_scores)
    
    print("--- Training Pipeline Completed Successfully! ---")
