from flask import Flask, request, redirect, url_for, Response, render_template, flash, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import tensorflow as tf
import json

app = Flask(__name__, template_folder='templates')

CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

import numpy as np # linear algebra

from zipfile import ZipFile
from pathlib import Path

# Data Visualization
import matplotlib.pyplot as plt

## 3.1 Preparing the Data
# safe each dataset into variable

rating = pd.read_csv('tourism_rating.csv')
place = pd.read_csv('tourism_with_id.csv')
user = pd.read_csv('user.csv')



"""## 3.2 Data Features Exploration"""

# Looking into data place 


# Drop unused column 

place = place.drop(['Unnamed: 11','Unnamed: 12'],axis=1)
place.head(2)

"""gausah dipake"""

# Show just Yogyakarta
#place = place[place['City']=='Yogyakarta']
#place.head(2)

place.loc[:, ['Time_Minutes']].mean(axis = 0)

place.info()

# looking into data rating

rating.head()

rating.info()

# change data rating so that it will contain destination rating pada of Yogyakarta

rating = pd.merge(rating, place[['Place_Id']], how='right', on='Place_Id')
rating.head()

# seeing the shape rating for Yogyakarta

rating.shape

# look into respondents' data 

user.head()

# change respondents' data into Yogyakarta destination visitors

user = pd.merge(user, rating[['User_Id']], how='right', on='User_Id').drop_duplicates().sort_values('User_Id')
user.head()

# looking respondents' dataset who gave rating to Yogyakarta destinations 

user.shape

"""# 4. Exploratory Data Analysis"""

# creating datafram that contains locations with most rating
top_10 = rating['Place_Id'].value_counts().reset_index()[0:10]
top_10 = pd.merge(top_10, place[['Place_Id','Place_Name']], how='left', left_on='index', right_on='Place_Id')
top_10

# changing the naming into English
place.Category[place.Category == 'Taman Hiburan'] = 'Amusement Park & Downtown Attractions'
place.Category[place.Category == 'Budaya'] = 'Culture'
place.Category[place.Category == 'Cagar Alam'] = 'National Park'
place.Category[place.Category == 'Taman Hiburan'] = 'Amusement Park'
place.Category[place.Category == 'Bahari'] = 'Marine Tourism'
place.Category[place.Category == 'Pusat Perbelanjaan'] = 'Shopping Center'

# filtering city origin of visitors
askot = user['Location'].apply(lambda x : x.split(',')[0])


"""# 5. Data Preparation for Modelling

## 5.1. Creating Copy for Data Rating
"""

# reading dataset for encoding
 
df = rating.copy()
df.head()

"""## 5.2. Encoding"""

def dict_encoder(col, data=df):

  # changing column of dataframe into list with unique value
  unique_val = data[col].unique().tolist()

  # enumerating column value of dataframe 
  val_to_val_encoded = {x: i for i, x in enumerate(unique_val)}

  # encoding process from numbers to column value of dataframe
  val_encoded_to_val = {i: x for i, x in enumerate(unique_val)}
  return val_to_val_encoded, val_encoded_to_val

# Encoding User_Id
user_to_user_encoded, user_encoded_to_user = dict_encoder('User_Id')

# Mapping User_Id into dataframe
df['user'] = df['User_Id'].map(user_to_user_encoded)

# Encoding Place_Id
place_to_place_encoded, place_encoded_to_place = dict_encoder('Place_Id')

# Mapping Place_Id into dataframe place
df['place'] = df['Place_Id'].map(place_to_place_encoded)

"""## 5.3. Looking into Data Modelling Overview"""

# getting length of user & place 
num_users, num_place = len(user_to_user_encoded), len(place_to_place_encoded)
 
# changing rating into float
df['Place_Ratings'] = df['Place_Ratings'].values.astype(np.float32)
 
# getting minimum and maximum rating
min_rating, max_rating = min(df['Place_Ratings']), max(df['Place_Ratings'])
 
print(f'Number of User: {num_users}, Number of Place: {num_place}, Min Rating: {min_rating}, Max Rating: {max_rating}')

# randomizing dataset
df = df.sample(frac=1, random_state=42)
df.head(2)


"""# 7. Prediction of 7 Recommended Destinations

## 7.1 DataFrame Preparation to Show the Recommendations
"""

# dataframe preparation
place_df = place[['Place_Id','Place_Name','Category','Rating','Price']]
place_df.columns = ['id','place_name','category','rating','price']
df = rating.copy()

"""## 7.2. User Example Preparation to Show Recommendations"""

# user sampling randomly
# user_id = request.args.get('user_id')
# user_id = int(user_id)



model = tf.keras.models.load_model('output')



@app.route('/recommend', methods=['POST'])
def rekomen():

    # user_id = df.User_Id.sample(1).iloc[0]
    req = request.get_json()
    user_id = req['user_id']

    place_visited_by_user = df[df.User_Id == user_id]

    # unvisited location data
    place_not_visited = place_df[~place_df['id'].isin(place_visited_by_user.Place_Id.values)]['id'] 
    place_not_visited = list(
        set(place_not_visited)
        .intersection(set(place_to_place_encoded.keys()))
    )
    
    place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
    user_encoder = user_to_user_encoded.get(user_id)
    user_place_array = np.hstack(
        ([[user_encoder]] * len(place_not_visited), place_not_visited)
    )

    # top 7 recommendations
    print(user_place_array)
    
    inputs = tf.cast(user_place_array, tf.int64)

    ratings = model.predict(inputs).flatten()
    top_ratings_indices = ratings.argsort()[-7:][::-1]
    recommended_place_ids = [
        place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices
    ]
    
    print('Recommendation list for: {}'.format('User ' + str(user_id)))
    print('===' * 15,'\n')
    print('----' * 15)
    print('Places with highest rating from users')
    print('----' * 15)
    
    top_place_user = (
        place_visited_by_user.sort_values(
            by = 'Place_Ratings',
            ascending=False
        )
        .head(5)
        .Place_Id.values
    )
    
    place_df_rows = place_df[place_df['id'].isin(top_place_user)]
    for row in place_df_rows.itertuples():
        print(row.place_name, ':', row.category)

    print('')
    print('----' * 15)
    print('Top 7 place recommendations')
    print('----' * 15)
    
    recommended_place = place_df[place_df['id'].isin(recommended_place_ids)]
    for row, i in zip(recommended_place.itertuples(), range(1,8)):
        print(i,'.', row.place_name, '\n    ', row.category, ',', 'Entrance Fee', row.price, ',', 'Rating', row.rating,'\n')

    print('==='*15)

    data = recommended_place.to_dict(orient='records')

    # Serialize the data to JSON
    # json_data = json.dumps(data)

    return jsonify(data)


# @app.route('/prediction')
# def index():
#     nama_tempat = request.args.get("nama_tempat")

# app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True)