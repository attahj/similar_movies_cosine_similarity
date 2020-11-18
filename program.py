import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

#read csv file
ratings = pd.read_csv('ratings.csv')


#pivot table for movie x user dataframe for ratings
mxu = pd.pivot_table(ratings, values = 'rating', index=['movieId'], columns=['userId'], fill_value=0)


#we will use cosine similarity with normalized dot product from sklearn
cos_sim = pd.DataFrame(cosine_similarity(mxu))
cos_sim.index = mxu.index.values.tolist()
cos_sim.columns = mxu.index.values.tolist()

#compute similar movies list for all movies (5)
rec_movies = list()
for i in mxu.index.unique().tolist():
    #searched movie and 5 similar movies
    rec_movies.append([i] + (((((cos_sim.loc[i])[1:]).sort_values(ascending =False)[1:6]).index.values.tolist())))

#create search-able dataframe of recommended movies
rec_movies = (pd.DataFrame(rec_movies)).set_index(0)

#Runs slow compared to all other things I would recommend commenting line 28-53 out to get the output for the final product then trying out this score prediction 
#Score prediction for unrated movies for each user
predict = mxu
for i in ratings.userId.unique().tolist():
    df = (pd.DataFrame(mxu[i]))
    
    #iterate through movies that havent been watched yet
    for j in ((df.loc[df[i] == 0.0]).index.values.tolist()):
        
        #similarity list for all movies with sim() score of .5 aka 2.0 or higher
        sim = (pd.DataFrame(cos_sim[j]))[cos_sim[j] >= 0.50]

        #ratings of these movies by the user 
        rating = df.loc[sim.index.values.tolist()]
        
        #if ratings for similar movies are empty leave the rating at 0
        if (rating[list(rating)[0]] == 0).all():
            continue;
        #else predict with pearson and update df
        else:
            rating = rating[rating[list(rating)[0]] !=0]
            sim = sim.loc[rating.index.values.tolist()]
            rating = rating[list(rating)[0]].tolist()
            sim = sim[list(sim)[0]].tolist()
            predictrating = [a*b for a,b in zip(rating,sim)]
            predictrating = sum(predictrating)/sum(sim)
            predict.at[j,i] = predictrating
            #print("row " + str(j) + " col " + str(i) + " has been updated to " + str(predictrating))
    
#recommended movies for each user
output = list()
for i in ratings.userId.unique().tolist():
    #break down movie profile list for each user
    df = (pd.DataFrame(mxu[i].sort_values(ascending=False))) 
    
    #create list of userID + list of movies to watch
    returnlist = list()
    #append userID
    returnlist.append(i)
    
    #iterate through watched movies
    for j in ((df.loc[df[i] != 0.0]).index.values.tolist()):
        #break-case at length of 5 movies 
        if len(returnlist) == 6: 
            output.append(returnlist)
            break;
        else:
            #look through list of recommended movies based on movie that is searched
            for k in (rec_movies.loc[j,:].values.tolist()):
                #break-case
                if len(returnlist) == 6:
                    break;
                #if not in watched list add to list for user to watch
                else:
                    if k in ((df.loc[df[i] == 0.0]).index.values.tolist()):
                        returnlist.append(k)

#generate txt file with output
textfile = ""
for i in output:
    textfile = textfile + " ".join(map(str,i)) +"\n"
    

#write to file    
file=open("output.txt", "w")
file.write(textfile)
file.close()        
