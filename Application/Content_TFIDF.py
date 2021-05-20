
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Get top_n similar indexes and their score respectively
def find_similar(tfidf_matrix, index, top_n = 11):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]] #if i != index
    return ([index for index in related_docs_indices][0:top_n],[idx for idx in sorted(cosine_similarities,reverse=True)][0:top_n])

def main(query):
        df = pd.read_csv("Netflix.csv")
       
        #------------------TD-IDF------------------
        # Vectorize data with tfidf
        vector = TfidfVectorizer(max_df=0.4,         # drop words that occur in more than X percent of documents
                                     min_df=1,       # only use words that appear at least X times
                                     stop_words='english', # remove stop words
                                     lowercase=True, # Convert everything to lower case
                                     use_idf=True,   # Use idf
                                     norm=u'l2',     # Normalization
                                     smooth_idf=True # Prevents divide-by-zero errors
                                    )
        tfidf = vector.fit_transform(df['movie_info'])

        # Find 10 most similar titles from user input using dot product
        request_transform = vector.transform([query])
        similarity = np.dot(request_transform,np.transpose(tfidf))
        x = np.array(similarity.toarray()[0])
        indices=np.argsort(x)[-10:][::-1]

        # Print 10 most similar titles with their id
        indlist = []
        for i in indices:
            indlist.append(i)
            print('id = {0:5d} - title = {1}'.format(i,df['title'].loc[i]))

        # Get id selection from user
        idnumb = -1
        while idnumb not in indlist:
            try:
                idnumb = int(input("Enter one id number from above:  "))
            except ValueError:
                print("Need a number not letter.")

        # Call def, get result and pop first
        result = find_similar(tfidf,idnumb)
        indexes, tdidfscore = result[0], result[1]
        indexes.pop(0)
        tdidfscore.pop(0)

        # Output 10 most similar movies
        movies_dict_tfidf = {}
        for x,y in zip(indexes,tdidfscore):
            movies_dict_tfidf[df.title[x]] = round(y,5)
        print("\nWe Reccomend")
        for i in movies_dict_tfidf.keys():
            print(i,movies_dict_tfidf[i])