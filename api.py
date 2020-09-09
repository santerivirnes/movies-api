from flask import Flask, json, jsonify, request
import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from flask_cors import CORS, cross_origin

#luodaan taulukko csv tiedostosta
df = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')


#napataan taulukosta kyseiset kolumnit
df = df[['Title','Genre','Director','Actors','Plot' ,'Poster', 'imdbID']]

#tehdään toinen taulukko myöhempää käyttöä varten
final_data = df[['Title', 'imdbID', 'Poster']]


# poistetaan pilkku näyttelijöiden nimien välistä ja napataan vain 3 ensimmäistä nimeä talteen
df['Actors'] = df['Actors'].map(lambda x: x.split(',')[:3])



# asetetaan genret listaan
df['Genre'] = df['Genre'].map(lambda x: x.lower().split(','))

# Poistetaan välit ohjaajien nimistä
df['Director'] = df['Director'].map(lambda x: x.split(' '))



# yhdistetään etu ja sukunimet, jotta saman etunimen kanssa ei tule sekaannuksia
for index, row in df.iterrows():
    row['Actors'] = [x.lower().replace(' ','') for x in row['Actors']]
    row['Director'] = ''.join(row['Director']).lower()



# Uusi kolumni johon juonen avain sanat paloitellaan
df['Key_words'] = ""


#Puretaan juoni avain sanoiksi ja talletaan ne edellä luotuun kolumniin
for index, row in df.iterrows():
    plot = row['Plot']
    
    # instantiating Rake, by default is uses english stopwords from NLTK, nämä kommentit säästetään tutorialista
    # and discard all puntuation characters, nämä kommentit säästetään tutorialista
    r = Rake()

    # extracting the words by passing the text, nämä kommentit säästetään tutorialista
    r.extract_keywords_from_text(plot)

    # getting the dictionary whith key words and their scores, nämä kommentit säästetään tutorialista
    key_words_dict_scores = r.get_word_degrees()
    
    # assigning the key words to the new column, nämä kommentit säästetään tutorialista
    row['Key_words'] = list(key_words_dict_scores.keys())

# Poistetaan juoni kolumni
df.drop(columns = ['Plot'], inplace = True)
# Ja otsikko kolumni
df.drop(columns = ['Title'], inplace = True)
df.set_index('imdbID', inplace = True)


# Talletetaan edellä purettujen kolumnien sanat yhteen kolumniin
df['bag_of_words'] = ''
columns = df.columns
for index, row in df.iterrows():
    words = ''
    for col in columns:
        if col != 'Director':
            words = words + ' '.join(row[col])+ ' '
        else:
            words = words + row[col]+ ' '
    row['bag_of_words'] = words
df.drop(columns = [col for col in df.columns if col!= 'bag_of_words'], inplace = True)




# Muodostetaan lasku matriisi
count = CountVectorizer()
count_matrix = count.fit_transform(df['bag_of_words'])



# Tehdään sarja elokuvien titteleistä niin että ne ovat numero järjestyksessä
# Listaa käytetään myöhemmin indeksien matchaamisessä
indices = pd.Series(df.index)
indices[:5]



# generating the cosine similarity matrix, nämä kommentit säästetään tutorialista
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# function that takes in movie title as input and returns the top 10 recommended movies, nämä kommentit säästetään tutorialista
def recommendations(title, cosine_sim = cosine_sim):
    
    recommended_movies = []

    if title in indices.values:
            

        # gettin the index of the movie that matches the imdbID, nämä kommentit säästetään tutorialista
        idx = indices[indices == title].index[0]
        
        # creating a Series with the similarity scores in descending order, nämä kommentit säästetään tutorialista
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

        
        
        # getting the indexes of the 10 most similar movies, nämä kommentit säästetään tutorialista
        top_10_indexes = list(score_series.iloc[1:11].index)
        

        # populating the list with the titles of the best 10 matching movies, nämä kommentit säästetään tutorialista
        for i in top_10_indexes:
            recommended_movies.append({'Title': list(final_data['Title'])[i], 'imdbID': list(final_data['imdbID'])[i], 'Poster': list(final_data['Poster'])[i]})
    
        return recommended_movies
    else:
        return recommended_movies



api = Flask(__name__)
CORS(api)

@api.route('/movies', methods=['GET', 'POST'])

def get_movies():
  data = request.get_json()
  return json.dumps(recommendations(data['id']))
  
  
  
if __name__ == '__main__':
    api.run()
