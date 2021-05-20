
from surprise import accuracy

class EvaluatedAlgorithm:
    
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name
       
    def Evaluate(self, evaluationData,n=10):
        metrics = {}
        print("Evaluating the Αccuracy of the System...")
        self.algorithm.fit(evaluationData.GetTrainSet())
        predictions = self.algorithm.test(evaluationData.GetTestSet())
        
        metrics["RMSE"] = self.RMSE(predictions)
        metrics["MAE"] = self.MAE(predictions)
        metrics["FCP"] = self.FCP(predictions)
        return metrics
    
    def MAE(self,predictions):
        return accuracy.mae(predictions, verbose=False)

    def RMSE(self,predictions):
        return accuracy.rmse(predictions, verbose=False)
       
    def FCP(self,predictions):
        return accuracy.fcp(predictions, verbose=False)
   
    def GetName(self):
        return self.name
    
    def GetAlgorithm(self):
        return self.algorithm   