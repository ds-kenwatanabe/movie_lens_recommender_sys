import pandas as pd

file_path = "/home/chris/PycharmProjects/recommender_movie_lens/ml-latest/movies.csv"
data = pd.read_csv(file_path)
print(data.head())
