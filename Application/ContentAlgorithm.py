
from surprise import AlgoBase
from surprise import PredictionImpossible
import math
import numpy as np
import heapq
import csv
from collections import defaultdict

class ContentAlgorithm(AlgoBase):
    moviesPath = 'Netflix.csv'
    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k
        
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        genres = self.getGenres()
        print("Computing content-based similarity matrix...")
            
        # Compute Genre similarity for every movie combination as a matrix
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))
        
        for thisRating in range(self.trainset.n_items):
            if (thisRating % 1000 == 0):
                print(thisRating, " of ", self.trainset.n_items)
            for otherRating in range(thisRating+1, self.trainset.n_items):
                thisMovieID = self.trainset.to_raw_iid(thisRating)
                otherMovieID = self.trainset.to_raw_iid(otherRating)
                genreSimilarity = self.computeGenreSimilarity(thisMovieID, otherMovieID, genres)
                
                self.similarities[thisRating, otherRating] = genreSimilarity 
                self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating]
                
        print("Done computing content-based similarity matrix.")
              
        return self

    def getGenres(self):
        genres = defaultdict(list)
        genreIDs = {}
        maxGenreID = 0
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)  #Skip header line
            for row in movieReader:
                movieID = row[0]
                genreList = row[2].split(',')
                genreIDList = []
                for genre in genreList:
                    if genre in genreIDs:
                        genreID = genreIDs[genre]
                    else:
                        genreID = maxGenreID
                        genreIDs[genre] = genreID
                        maxGenreID += 1
                    genreIDList.append(genreID)
                genres[movieID] = genreIDList
        # Convert integer-encoded genre lists to bitfields that we can treat as vectors
        for (movieID, genreIDList) in genres.items():
            bitfield = [0] * maxGenreID
            for genreID in genreIDList:
                bitfield[genreID] = 1
            genres[movieID] = bitfield            
        return genres
    
    def computeGenreSimilarity(self, movie1, movie2, genres):
        genres1 = genres[movie1]
        genres2 = genres[movie2]
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(genres1)):
            x = genres1[i]
            y = genres2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        
        return sumxy/math.sqrt(sumxx*sumyy)   #Cosine similarity
    
    def estimate(self, u, i):
        # In case the user or the movie is not in Database 
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        
        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[u]:
            genreSimilarity = self.similarities[i,rating[0]]
            neighbors.append( (genreSimilarity, rating[1]) )
        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        
        # Compute average sim score of K neighbors weighted by user ratings
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating
        
        # In case the system can not find similar item    
        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')
        
        predictedRating= weightedSum / simTotal
              
        return (predictedRating,)