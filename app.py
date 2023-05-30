import streamlit as st
import pandas as pd
import joblib

scaler = joblib.load('minmaxScaler.joblib')

model = joblib.load('decisionTree.joblib')

# page setting
st.set_page_config(page_title="Spotify songs skips Predictor App",  layout="centered")

# page header
st.title(f"Spotify songs skips Predictor App")

#features = [end_trackdone, start_trackdone, start_fwdbtn, end_fwdbtn, end_backbtn, no_pause_before_play, long_pause_before_play,
#            valence, vector_6, vector_5, duration, dyn_range_mean, vector_1, organism, energy, vector_2, us_popularity,
#            bounciness, short_pause_before_play, beat_strength]

with st.form("Prediction_form"):
    # form header
    st.header("Enter the specifications:")
    # input elements
    end_trackdone = st.selectbox(label="end_trackdone: ", options=[0.0, 1.0])
    start_trackdone = st.selectbox(label="start_trackdone: ", options=[0.0, 1.0])
    start_fwdbtn = st.selectbox(label="start_fwdbtn: ", options=[0.0, 1.0])
    end_fwdbtn = st.selectbox(label="end_fwdbtn: ", options=[0.0, 1.0])
    end_backbtn = st.selectbox(label="end_backbtn: ", options=[0.0, 1.0])
    no_pause_before_play = st.selectbox(label="no_pause_before_play : ", options=[0.0, 1.0])
    long_pause_before_play = st.selectbox(label="long_pause_before_play: ", options=[0.0, 1.0])
    valence = st.slider(label="valence: ", min_value=0.0, max_value=1.00, step=0.001)
    vector_6 = st.slider(label="vector_6: ", min_value=-0.7, max_value=0.30, step=0.001)
    vector_5 = st.slider(label="vector_5: ", min_value=0.20, max_value=0.40, step=0.001)
    duration = st.slider(label="duration : ", min_value=30.0, max_value=1780.00, step=1.0)
    dyn_range_mean = st.slider(label="dyn_range_mean: ", min_value=1.0, max_value=16.0, step=0.1)
    vector_1 = st.slider(label="vector_1: ", min_value=0.0, max_value=0.60, step=0.01)
    organism = st.slider(label="organism: ", min_value=0.0, max_value=1.00, step=0.01)
    energy = st.slider(label="energy: ", min_value=0.0, max_value=1.00, step=0.01)
    vector_2 = st.slider(label="vector_2: ", min_value=0.0, max_value=0.50, step=0.01)
    us_popularity = st.slider(label="us_popularity: ", min_value=90.0, max_value=100.00, step=0.01)
    bounciness = st.slider(label="bounciness: ", min_value=0.0, max_value=5.00, step=0.01)
    short_pause_before_play = st.selectbox(label="short_pause_before_play: ", options=[0.0, 1.0])
    beat_strength = st.slider(label="beat_strength: ", min_value=0.0, max_value=1.00, step=0.01)
    
    # submitt values
    submit_values = st.form_submit_button("Predict")

if submit_values:

    cols_to_scale = {'valence':valence, 'vector_6':vector_6, 'vector_5':vector_5,'duration':duration, 
                    'dyn_range_mean':dyn_range_mean, 'vector_1':vector_1, 'organism':organism, 'energy': energy,
                    'vector_2':vector_2, 'us_popularity':us_popularity,'bounciness':bounciness, 
                    'beat_strength':beat_strength}
    

    # create scaling input dataframe
    scaling_input = pd.DataFrame(cols_to_scale, index=[1])

    scaling = scaler.transform(scaling_input)

    scaled = pd.DataFrame(scaling,columns=cols_to_scale)

    
    input_dict = {
        'end_trackdone': end_trackdone, 'start_trackdone':start_trackdone, 'start_fwdbtn':start_fwdbtn,
        'end_fwdbtn':end_fwdbtn, 'end_backbtn':end_backbtn, 'no_pause_before_play':no_pause_before_play,
        'long_pause_before_play':long_pause_before_play, 'valence': scaled.valence[0], 'vector_6':scaled.vector_6[0], 
        'vector_5':scaled.vector_5[0], 'duration':scaled.duration[0], 'dyn_range_mean': scaled.dyn_range_mean[0],
        'vector_1': scaled.vector_1[0],'organism': scaled.organism[0], 'energy': scaled.energy[0],
        'vector_2': scaled.vector_2[0], 'us_popularity': scaled.us_popularity[0] , 'bounciness': scaled.bounciness[0], 
        'short_pause_before_play': short_pause_before_play, 'beat_strength':scaled.beat_strength[0] }
    
    # create input dataframe
    input_dataframe = pd.DataFrame(input_dict, index=[1])

    
    # make predictions
    prediction = model.predict(input_dataframe)

    if prediction == 0:
        value = "wont't skip"
    elif prediction == 1:
        value = "will_skip"
    
    # output header
    st.header("Here is the prediction: ")
    # output results
    st.success(f'The user {value} the song')
    # balloons..
    st.balloons()