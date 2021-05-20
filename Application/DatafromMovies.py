
import csv
from surprise import Dataset
from surprise import Reader

class DatafromMovies:

    movieID_to_name = {}
    ratingsPath = 'SurpRatings.csv'
    moviesPath = 'Netflix.csv'
    
    def MoviesratingsDataset(self):
        ratingsDataset = 0
        self.movieID_to_name = {}

        reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
        ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
                movieReader = csv.reader(csvfile)
                next(movieReader)
                for row in movieReader:
                    movieID = row[0]
                    movieName = row[1]
                    self.movieID_to_name[movieID] = movieName

        return ratingsDataset

    def getMovieName(self, movieID):
        if movieID in self.movieID_to_name:
            return self.movieID_to_name[movieID]
        else:
            return ""