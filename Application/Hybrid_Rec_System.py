
from DatafromMovies import DatafromMovies
from surprise import KNNBasic
from ContentAlgorithm import ContentAlgorithm
from HybridAlgorithm import HybridAlgorithm
from Evaluator import Evaluator

def LoadMoviesData():
    mv = DatafromMovies()
    print("Loading Movie Ratings...")
    data = mv.MoviesratingsDataset()
    return (mv, data)

def main(testSubject):
    # Load dataset for Recommender Algorithms
    (mv, evaluationData)= LoadMoviesData()
    
    print("Construct an Evaluator to evaluate Algorithms")
    evaluator = Evaluator(evaluationData)
    
    # Content KNN
    ContentKNN = ContentAlgorithm()
    
    # User - based KNN
    UserKNN = KNNBasic(sim_options = {'name': 'pearson', 'user_based': True})
    
    # Item - based KNN
    ItemKNN = KNNBasic(sim_options = {'name': 'pearson', 'user_based': False})
       
    # Combination 
    Hybrid = HybridAlgorithm([UserKNN,ItemKNN,ContentKNN], [0.4, 0.5, 0.1])
    
    evaluator.AddAlgorithm(Hybrid, "Hybrid")
   
    # Enable if you want to evaluate Collaborative Filtering
    # evaluator.AddAlgorithm(ItemKNN, "ItemKNN")
    # evaluator.AddAlgorithm(UserKNN , "UserKNN")
  
    # Reccomendations
    evaluator.SampleTopNRecs(mv,testSubject)
    
    # Evaluate Hybrid!
    evaluator.Evaluate()