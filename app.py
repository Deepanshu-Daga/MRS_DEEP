
import streamlit as st
import pandas as pd
import pickle
import time

st. set_page_config(layout="wide")
st.title("Deep's Movie Recommendation")
movies_dict = pickle.load(open('movie_dict.pkl','rb'))
new_df = pd.DataFrame(movies_dict)
movies_names = new_df['title'].values

movie_input = st.selectbox('Which movie do you like ?', movies_names)
start = time.time()
# if st.button('Recommend'):     # if button is required.



# From second run onwards Approach Two

# # now we want the movie image via the movie_id we saved from imdb
# # api fx
# def fetch_image_path(m_id):
#     import requests
#     url = "https://api.themoviedb.org/3/movie/{}".format(m_id)
#     headers = {
#         "accept": "application/json",
#         "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiNGE5MTYzYmI5ZjFlZDA1YjBlYWU5YWFhYWIzMGU0NCIsInN1YiI6IjYzZWI2ZDE0Njk5ZmI3MDA5NmFkMWQzMiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.9JwBNZoA7S0sAmaAAaU7SBSgWgjrDyTkNee8L3XJkmc"
#     }
#     response = requests.get(url, headers=headers)
#     my_data = response.json()

#     # print(response.text)
#     return ('https://image.tmdb.org/t/p/original' + my_data['poster_path'])



# cosine_similarity_matrix = pickle.load(open('cosine_similarity_matrix.pkl','rb'))

# # To create our recomendation fx we create a fx

# def my_recommendation(movie_input):
#     movie_index = new_df[new_df['title'] == movie_input].index[0]
#     recomend_movie = sorted(enumerate(cosine_similarity_matrix[movie_index]), reverse=True, key=lambda x: x[1])[1:6]

#     image_path = []
#     recommended_movies = []
#     st.write('Recommended Movies For You : ')
#     for i in recomend_movie:
#         recommended_movies.append(new_df.iloc[i[0]]['title'])
#         image_path.append(fetch_image_path(new_df.iloc[i[0]]['movie_id']))
#         # st.write(new_df.iloc[i[0]]['title'])
#     return (recommended_movies, image_path)


# r_movies,path = my_recommendation(movie_input)


# import streamlit as st

# col1, col2, col3 ,col4, col5= st.columns(5)

# with col1:
#    st.text(r_movies[0])
#    st.image(path[0])

# with col2:
#    st.text(r_movies[1])
#    st.image(path[1])

# with col3:
#    st.text(r_movies[2])
#    st.image(path[2])

# with col4:
#    st.text(r_movies[3])
#    st.image(path[3])

# with col5:
#    st.text(r_movies[4])
#    st.image(path[4])

# st.snow()
# end = time.time()
# st.write((end - start) * (1000), "ms")




# Approach number 1 and first time run to create cosine_similarity_matrix
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 6000 , stop_words= 'english')
movie_vectors = cv.fit_transform(new_df['tags']).toarray()          # Learn the vocabulary dictionary and return document-term matrix thus converting it to array

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(movie_vectors)
cos_var = cosine_similarity(movie_vectors)           #cosine radian angle diff with all movies with all other movies matrix of ((4806, 4806))
pickle.dump(cos_var,open('cosine_similarity_matrix.pkl','wb'))
# To create our recomendation fx we create a fx

cosine_similarity_matrix = pickle.load(open('cosine_similarity_matrix.pkl','rb'))

def my_recommendation(movie_input):
    movie_index = new_df[new_df['title'] == movie_input].index[0]
    recomend_movie = sorted(enumerate(cosine_similarity_matrix[movie_index]), reverse=True, key=lambda x: x[1])[1:6]

    image_path = []
    recommended_movies = []
    st.write('Recommended Movies For You : ')
    for i in recomend_movie:
        recommended_movies.append(new_df.iloc[i[0]]['title'])
        image_path.append(fetch_image_path(new_df.iloc[i[0]]['movie_id']))
        # st.write(new_df.iloc[i[0]]['title'])
    return (recommended_movies, image_path)


r_movies,path = my_recommendation(movie_input)


import streamlit as st

col1, col2, col3 ,col4, col5= st.columns(5)

with col1:
   st.text(r_movies[0])
   st.image(path[0])

with col2:
   st.text(r_movies[1])
   st.image(path[1])

with col3:
   st.text(r_movies[2])
   st.image(path[2])

with col4:
   st.text(r_movies[3])
   st.image(path[3])

with col5:
   st.text(r_movies[4])
   st.image(path[4])

st.snow()
end = time.time()
st.write((end - start) * (1000), "ms")
