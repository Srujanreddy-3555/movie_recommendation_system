Movie Recommendation System
This project demonstrates a Movie Recommendation System using Python, leveraging datasets that include movie ratings and metadata. It provides recommendations based on genres, similar users, and similarity measures such as cosine similarity.

Features
Exploratory Data Analysis: Analyzing movie and rating data to gain insights about unique users, movies, and genres.
Genre-based Recommendations: Recommend movies by genres, filtered by minimum review thresholds.
Title-based Recommendations: Recommend movies similar to a selected movie title, based on genre similarity.
User-based Recommendations: Recommend movies based on user similarity using cosine similarity.
Interactive Widgets: Provide an interactive widget interface to get movie recommendations by genre.
Installation

Install dependencies: Ensure that you have Python 3.8+ installed. Install the necessary dependencies by running:

bash
Copy code
pip install -r requirements.txt
Jupyter Notebook: To run the interactive recommendation system using Jupyter widgets, install Jupyter and ipywidgets:

bash
Copy code
pip install notebook ipywidgets
Usage
1. Data Exploration
Explore the movie and rating data with the provided script:

python
Copy code
import pandas as pd

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Get an overview of the datasets
print(ratings.shape)
print(movies.shape)
print(ratings.head())
print(movies.head())

# Descriptive statistics
print(ratings.describe())

# Number of unique users and movies
unique_users = ratings['userId'].nunique()
unique_movies = ratings['movieId'].nunique()
print("Number of unique users:", unique_users)
print("Number of unique movies:", unique_movies)
2. Genre-based Recommendation
You can get movie recommendations based on a specific genre with a threshold of reviews and top N recommendations.

python
Copy code
import pandas as pd

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

genre = input('Enter the genre: ')
threshold = int(input("Enter the minimum review threshold: "))
N = int(input("Enter the number of recommendations: "))

# Generate and print recommendations
genre_movies = movies[movies['genres'].str.contains(genre)]
review_counts = ratings['movieId'].value_counts().rename('review_count')
genre_movies = genre_movies.merge(review_counts, left_on='movieId', right_index=True)
genre_movies = genre_movies[genre_movies['review_count'] >= threshold]

average_ratings = ratings.groupby('movieId')['rating'].mean().rename('average_rating')
genre_movies = genre_movies.merge(average_ratings, left_on='movieId', right_index=True)
movies_sorted = genre_movies.sort_values(by='average_rating', ascending=False)

recommended_movies = movies_sorted.head(N)
print(recommended_movies[['movieId', 'title', 'average_rating']])
3. Title-based Movie Recommendation
Get movie recommendations based on a selected movie title by genre similarity.

python
Copy code
import pandas as pd

movies = pd.read_csv('movies.csv')

movie_title = input('Enter the Movie Title: ')
N = int(input('Enter number of recommendations: '))

# Check if movie exists and find similar movies based on genre
if movie_title in movies['title'].values:
    selected_movie = movies[movies['title'] == movie_title].iloc[0]
    selected_movie_genres = selected_movie['genres']

    similar_movies = movies[movies['genres'].apply(lambda x: any(genre in x for genre in selected_movie_genres))]
    similar_movies['similarity_score'] = similar_movies.apply(lambda row: sum(genre in selected_movie_genres for genre in row['genres']), axis=1)
    
    sorted_movies = similar_movies.sort_values(by='similarity_score', ascending=False)
    recommended_movies = sorted_movies.head(N)
    print(recommended_movies[['movieId', 'title', 'similarity_score']])
else:
    print("Movie title not found in the dataset.")
4. User-based Movie Recommendation using Cosine Similarity
This module finds movies rated highly by users similar to a given user based on cosine similarity.

python
Copy code
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv('ratings.csv')
user_id = int(input('UserID: '))
N = int(input('Num Recommendations: '))
K = int(input('Threshold: '))

# Calculate similarity matrix and generate recommendations
similarity_matrix = cosine_similarity(ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0))
similar_users_indices = similarity_matrix[user_id-1].argsort()[::-1][1:K+1]
movies_rated_by_similar_users = ratings[ratings['userId'].isin(similar_users_indices + 1)]

average_ratings = movies_rated_by_similar_users.groupby('movieId')['rating'].mean().rename('average_rating')
recommended_movies = average_ratings.sort_values(ascending=False).head(N)

print(recommended_movies)
5. Interactive Widget-based Recommendation
For a more interactive experience, the system allows for recommendations via widgets. Launch Jupyter Notebook and use the following code:

python
Copy code
import pandas as pd
import ipywidgets as widgets
from IPython.display import display

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

def recommend_movies(genre, threshold, N):
    genre_movies = movies[movies['genres'].str.contains(genre)]
    review_counts = ratings['movieId'].value_counts().rename('review_count')
    genre_movies = genre_movies.merge(review_counts, left_on='movieId', right_index=True)
    genre_movies = genre_movies[genre_movies['review_count'] >= threshold]
    average_ratings = ratings.groupby('movieId')['rating'].mean().rename('average_rating')
    genre_movies = genre_movies.merge(average_ratings, left_on='movieId', right_index=True)
    return genre_movies.sort_values(by='average_rating', ascending=False).head(N)

genre_widget = widgets.Text(description='Genre:')
threshold_widget = widgets.IntSlider(description='Minimum review threshold:', min=0, max=500, step=10)
N_widget = widgets.IntText(description='Number of recommendations:', min=1, max=10)

button = widgets.Button(description='Recommend Movies')
output = widgets.Output()

def on_button_clicked(b):
    with output:
        output.clear_output()
        recommended_movies = recommend_movies(genre_widget.value, threshold_widget.value, N_widget.value)
        display(recommended_movies)

button.on_click(on_button_clicked)
display(genre_widget, threshold_widget, N_widget, button, output)
Data
ratings.csv: Contains user ratings for movies.
movies.csv: Contains movie metadata, including titles and genres.
Requirements
Python 3.8+
Pandas
scikit-learn
ipywidgets (for interactive features)
Jupyter Notebook (for interactive features)
To install all dependencies, run:

bash
Copy code
pip install pandas scikit-learn ipywidgets
This README provides a comprehensive overview of the project, including how to use different functionalities of the Movie Recommendation System.
