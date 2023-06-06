<p align="center">
  <img src="file:///C:/Users/Tuhin/Desktop/spotify.png">
</p>


# Project - Creating a machine-learning model to predict Song-Skips on Sequential User and Acoustic-Features

### Abstract
The main purposes of this project are to find out the important of input variables i.e acoustic features that affect mostly in the songs
skip prediction and make a predictive application by Machine-Learning techiques for the client to facilitate the comprehension of user behaviour.

### Background of Understanding the Problem
The mucical app spotify has provided two datasets. One dataset contains data related to user behaviour and other one contains data related acoustic features of all individual 
musical tracks. Now we have to find out the the most relavant user-behaviour and acoustic features which contribute to the skip of a particular track so that the client i.e spotify be able to 
see the impact of the input features on skipping of a song and thus they can improve their service in various ways such as by accurately predicting song skips, Spotify can better understand user
preferences and tailor personalized recommendations, by identifying songs that are more likely to be skipped, Spotify can optimize playlist sequencing and ensure a more enjoyable listening experience.
Thus finally they are able to meet their business goals.


### Data Details

*	log-mini.csv (df1) has 167880 rows and 21 columns
*	tf-mini.csv (df2) has 50704 rows and 30 columns
*	Data has no duplicates.
*	Data has missing values.

### Data prepeocessing

• First, we'll rename 'track_id_clean' feature of df1 as 'track_id' then merge the two datasets (i.e df1 & df2) on common feature column 'track-id'


```python
# Renaming the column track id
df1 = df1.rename(columns={'track_id_clean':'track_id'})

# merging the two dataset
df = pd.merge(df1, df2, on='track_id')
```

• Then we create another column named 'skipped' by combining 'skip1','skip2','skip3' through multiplication

```python
df["skipped"] = df["skip_1"]*df["skip_2"]*df["skip_3"]
df.drop(["skip_1", "skip_2", "skip_3", "not_skipped"], axis=1, inplace=True)
```

### Exploratory Data Analysis (EDA)

In EDA, we have examined a number of columns to learn more about the provided dataset.

## EDA of Input variables with binary features

![Example Image](https://drive.google.com/drive/u/0/folders/18lN8w6pmCyXlInbcBSl-Upf8SqCaO215) 

## EDA of Input variables with multiple features

![Example Image](https://github.com/tuhin3101/Spotify-project/blob/main/EDA_mul_features.gif) 


## Observations from EDA
1. Maximum songs are present in the first three sessions and minimum in last three sessions. And the songs belonging to 1st session are skipped least as compared to others.
2. Maximum skips happened for those songs which are present in the sessions starting from session 4 upto session 10 then no of skips decreased along with the no of songs.
3. Maximum count of sessions length is 20 which means users generally listen to music for 20 mins
4. The users do not seek forward or backward through the song
5. The users do not listen to music much at late nights or early in the morning. The peak listening hours are 1pm to 5pm in the afternoon
6. Maximum songs were listened in 15-07-2018
7. The maximum number of songs fall under the context user_collection. But songs present in editorial_playlist and catalog are less skipped as compared to user_collection.
8. The 'trackdone' category has maximum no of songs that are not skipped after playing and 'fwdbtn' category has most songs skipped. It indicates that the users skipped the song when they try to play the songs by forward option but they didn't skip when the songs are played in general order.
9. Maximum songs are released in 2018
10. Maximum songs fall under the category key-1 and minimum under the category key-3
11. The most common time signature in music is 4/4



## Outlier detection and outlier capping

 I used sns.boxplot() to identify the outliers from the following columns 
```python
cols = ['acousticness','beat_strength', 'bounciness', 'danceability', 'dyn_range_mean',
       'energy', 'flatness', 'instrumentalness', 'liveness', 'loudness','mechanism',
       'organism', 'speechiness', 'tempo', 'valence', 'acoustic_vector_0', 'acoustic_vector_1',
       'acoustic_vector_2', 'acoustic_vector_3', 'acoustic_vector_4',
       'acoustic_vector_5', 'acoustic_vector_6', 'acoustic_vector_7']
```
 Then applied capping i.e seting the upper and lower limit with threshold values.

```python

for i in cols:
    q1 = np.percentile(df[i],25)
    q3 = np.percentile(df[i],75)
    iqr = q3 - q1
    lw_lm = q1 - 1.5*iqr
    up_lm = q3 + 1.5*iqr
    df[i] = np.where(df[i]>up_lm, up_lm, np.where(df[i]<lw_lm, lw_lm, df[i]))

```
## Encoding

 We applied one hot encoding on the following columns 
```python
df = pd.get_dummies(df,columns=['context_type','hist_user_behavior_reason_start','hist_user_behavior_reason_end'])

```
 Then we applied label encoding on 'mode'
```python
encoder = LabelEncoder()
df['mode'] = encoder.fit_transform(df['mode'])
```
## Feature selection

From sklearn.feature_selection we imported mutual_info_regression in order to get top 20 features

![Example Image](https://github.com/tuhin3101/Spotify-project/blob/main/mi_scores.png) 

![Example Image](https://github.com/tuhin3101/Spotify-project/blob/main/top_20_features.png) 

## Mapping the column names for simplification

```python
col_map = {
    
 'hist_user_behavior_reason_end_trackdone': 'end_trackdone',
 'hist_user_behavior_reason_start_trackdone': 'start_trackdone',
 'hist_user_behavior_reason_start_fwdbtn': 'start_fwdbtn',
 'hist_user_behavior_reason_end_fwdbtn': 'end_fwdbtn' ,
 'hist_user_behavior_reason_end_backbtn': 'end_backbtn',
 'no_pause_before_play': 'no_pause_before_play',
 'long_pause_before_play': 'long_pause_before_play',
 'valence': 'valence',
 'acoustic_vector_6': 'vector_6',
 'acoustic_vector_5': 'vector_5',
 'duration':'duration',
 'dyn_range_mean': 'dyn_range_mean',
 'acoustic_vector_1': 'vector_1',
 'organism': 'organism',
 'energy': 'energy',
 'acoustic_vector_2': 'vector_2',
 'us_popularity_estimate': 'us_popularity',
 'bounciness': 'bounciness',
 'short_pause_before_play': 'short_pause_before_play',
 'beat_strength': 'beat_strength'
    
}
```
### Applying minmax scaling scaling on numerical variables with large unique values

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X[['valence', 'vector_6', 'vector_5','duration', 'dyn_range_mean', 'vector_1',
   'organism', 'energy', 'vector_2', 'us_popularity','bounciness',
   'beat_strength']] = scaler.fit_transform(X[['valence', 'vector_6', 'vector_5', 
                    'duration', 'dyn_range_mean', 'vector_1','organism', 'energy', 'vector_2', 'us_popularity',
                    'bounciness', 'beat_strength']])
```
 - Here X is input features

## Saving the scaler by joblib.dump

joblib.dump(scaler,'minmaxScaler.joblib')

## Modeling

### Train test split

```python
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
```
### Appying DecisionTree classifier
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  accuracy_score
dtc = DecisionTreeClassifier(max_depth=5,
			     criterion="gini",
			     random_state=100,
                             min_samples_split=2,
			     min_samples_leaf=2)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
accuracy_score(y_test, y_pred)
```

#### Training accuracy is - 0.874262866333095

#### Testing accuracy is - 0.8765487252799619


## Saving the model by joblib.dump

joblib.dump(dtc,'decisionTree.joblib')

