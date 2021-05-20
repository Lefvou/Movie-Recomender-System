
from SetBuilder import SetBuilder
from EvaluatedAlgorithm import EvaluatedAlgorithm

class Evaluator:
    
    algorithms = []
    
    def __init__(self, dataset):
        ed = SetBuilder(dataset)
        self.dataset = ed
    def AddAlgorithm(self, algorithm, name):
        alg = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(alg)
        
    def Evaluate(self):
        results = {}
        for algorithm in self.algorithms:
            print("Evaluating ", algorithm.GetName(), "...")
            results[algorithm.GetName()] = algorithm.Evaluate(self.dataset)

        # Print Results
        print("\n")
        print("{:<10} {:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE", "FCP"))
        for (name, metrics) in results.items():
            print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"],metrics["FCP"]))
       
        print("\n")
        print("RMSE:   Root Mean Squared Error. Lower values mean better accuracy.")
        print("MAE:    Mean Absolute Error. Lower values mean better accuracy.")
        print("FCP:    The proportion of well ranked items pairs. Higher values mean better accuracy.")
        
    def SampleTopNRecs(self, mv, testSubject, k=10):
        
        for algo in self.algorithms:
            print("\nUsing recommender ", algo.GetName())
            
            print("\nBuilding recommendation model...")
            trainSet = self.dataset.GetFullTrainSet()
            algo.GetAlgorithm().fit(trainSet)
            
            print("Computing recommendations...")
            testSet = self.dataset.GetAntiTestSetForUser(testSubject)
            # Make predictions for unseen items
            predictions = algo.GetAlgorithm().test(testSet)
            recommendations = []
                        
            for a,movieID,b,estimatedRating,c in predictions:
                recommendations.append((movieID, estimatedRating))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            print ("\nWe recommend:")
            for ratings in recommendations[:10]:
                print(mv.getMovieName(ratings[0]), ratings[1])