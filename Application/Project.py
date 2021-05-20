
import Content_TFIDF
import Hybrid_Rec_System 

Entrance= input("Have you already an account ?? ").lower()

while Entrance not in ["yes","y","no","n"]:
    Entrance= input("Have you already an account ?? ").lower()


if Entrance=="yes" or Entrance=="y":
    userID = input("Insert your userID in order to find reccomendations: ")
    Hybrid_Rec_System.main(userID)


elif Entrance=="no" or Entrance=="n":    
    query = input("Insert a movie you would like to see in order to find reccomendations: ")
    Content_TFIDF.main(query)
    
print("\nGoodBye!!!")