
from surprise.model_selection import train_test_split

class SetBuilder:
    
    def __init__(self, data):
        # Build a full train set
        self.fullTrainSet = data.build_full_trainset()
        # Build a 80/20 train/test split for measuring accuracy
        self.trainSet, self.testSet = train_test_split(data, test_size=.2, random_state=1)
    
    def GetFullTrainSet(self):
        return self.fullTrainSet
    # Create a list with unwatched from testuser
    def GetAntiTestSetForUser(self, testSubject):
        trainset = self.fullTrainSet
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(str(testSubject))
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                                 i in trainset.all_items() if
                                 i not in user_items]
        return anti_testset

    def GetTrainSet(self):
        return self.trainSet
    
    def GetTestSet(self):
        return self.testSet      