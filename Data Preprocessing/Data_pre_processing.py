import random
import pandas as pd 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Filtering_Text
def lemmatize_words(text):
            # Creates a set with all stop words
            stop = set(stopwords.words('english'))
        
            # Creates a list with punctuation
            punctuation = list(string.punctuation)
            stop.update(punctuation)
            # Function to remove stop words from df['movie info']
            lemmatizer = WordNetLemmatizer()
            final_text = []
            for i in text.split():
                if i.strip().lower() not in stop:
                    word = lemmatizer.lemmatize(i.strip())
                    final_text.append(word.lower())
            return " ".join(final_text)
#Load Data
netflix = pd.read_csv("kaggle.csv")

#Check for duplicated values
print(netflix.duplicated().sum())

#Data Cleaning.... We keep only the neccessary information
netflix = netflix.drop(columns=['director', 'country','date_added',"release_year","rating","duration","type","cast"])

# Combine all columns in order to find similarities on all of them
# We can add cast,country,date_added,release_year,rating,duration as future extection
netflix['movie_info'] = netflix['title'] + ' '+ netflix['listed_in']+ ' ' + netflix['description']
# Apply function
netflix.movie_info = netflix.movie_info.apply(lemmatize_words)
#Reccomendation file 
netflix.to_csv("Netflix.csv",index=False)

# Creation of Ratings
user={}
movies=list(netflix['show_id'])

for i in range(1,15000): #Number of users
    user[i]={}
for i in user.keys():
    for j in range(random.randint(80,150)): #random number of movies
        user[i][random.choice(movies)]=random.randint(1,5)
lista=[]
for i in user.keys():
    for j in user[i].keys():
        lista.append([i,j,user[i][j]])

df=pd.DataFrame(data=lista,columns=("User","Item","Rating"))

df.to_csv("SurpRatings.csv",index=False)