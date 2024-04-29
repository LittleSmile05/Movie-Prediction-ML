from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

movies_data = pd.read_csv('./movies.csv')
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director','homepage']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + \
                    movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + \
                    movies_data['director'] + ' ' + movies_data['homepage'].fillna('')
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']
    find_close_match = difflib.get_close_matches(movie_name, movies_data['title'])
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data['title'] == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    recommended_movies = []
    for i, movie in enumerate(sorted_similar_movies):
        if i < 5:
            index = movie[0]
            title_from_index = movies_data[movies_data['index'] == index]['title'].values[0]
            homepage_from_index = movies_data[movies_data['index'] == index]['homepage'].values[0]
            recommended_movies.append({'title': title_from_index, 'homepage': homepage_from_index})
    return render_template('recommendations.html', movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
