
import pandas as pd
from DatafromMovies import DatafromMovies
from surprise import KNNBasic
from surprise import KNNWithZScore
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise.model_selection import cross_validate

mv = DatafromMovies()
print("Loading Movie Ratings...")
data = mv.MoviesratingsDataset()
    
benchmark = []
# Iterate over all algorithms
for algorithm in [ KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore()]:
    # Perform cross validation
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
    
pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse') 