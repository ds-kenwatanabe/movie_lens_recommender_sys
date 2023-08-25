import pandas as pd

file_path = "/home/chris/PycharmProjects/movie_lens_recommender_sys/ml-25m/movies.csv"
"""def get_movie_ids(file_path: str):
    data = pd.read_csv(file_path)
    try:
        movie = input("Type the name of the movie: ")
        if movie in data:
            print(movie[])
"""

data = pd.read_csv(file_path)
print(data.head())
