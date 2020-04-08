from flask import Flask, request, render_template, jsonify
#import json
#from dicttoxml import dicttoxml
import spotipy
#import pandas as pd
import numpy as np
import pickle
from spotipy.oauth2 import SpotifyClientCredentials #To access authorised Spotify data

app = Flask(__name__)

def returnTrack(artist, track):
    client_id = '4933cbccbac34e74a7162ef1f2930dc0'
    client_secret = 'af1e13b5b65b4945b18447096b0be14b'
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)  # spotify object to access API
    artist = artist  # chosen artist
    track = track
    result = sp.search(q='artist:' + artist + ' track:' + track, type='track')  # search query
    item = result['tracks']['items'][0]
    return item

def returnFeatures(item):
    client_id = '4933cbccbac34e74a7162ef1f2930dc0'
    client_secret = 'af1e13b5b65b4945b18447096b0be14b'
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)  # spotify object to access API
    id = item['id']
    #track = item['name']
    #artist = item['artists'][0]['name']
    result = sp.audio_features(id)
    dic_df = {}
    #dic_df['Track'] = track
    #dic_df['Artist'] = artist
    dic_df['Acousticness'] = result[0]['acousticness']
    dic_df['Danceability'] = result[0]['danceability']
    dic_df['Energy'] = result[0]['energy']
    dic_df['Key'] = result[0]['key']
    dic_df['Instrumentalness'] = result[0]['instrumentalness']
    dic_df['Liveness'] = result[0]['liveness']
    loudness = result[0]['loudness']
    if loudness < -30:
        loudness = 1
    elif loudness >= -30 and loudness < -20:
        loudness = 2
    elif loudness >= -20 and loudness < -10:
        loudness = 3
    else:
        loudness = 4
    dic_df['Loudness'] = loudness
    dic_df['Speechiness'] = result[0]['speechiness']
    dic_df['Tempo'] = result[0]['tempo']
    dic_df['Mode'] = result[0]['mode']
    duration = result[0]['duration_ms']
    duration = duration / 1000
    dic_df['Duration(s)'] = duration
    dic_df['Time_signature'] = result[0]['time_signature']
    dic_df['Valence'] = result[0]['valence']
    scale_keys = ['Acousticness','Danceability','Energy','Instrumentalness','Liveness','Speechiness','Valence']
    for key in scale_keys:
        dic_df[key] = round(dic_df[key] * 10, 1)

    return dic_df

def returnResult(artist, track):
    item = returnTrack(artist, track)
    result = returnFeatures(item)
    return result

def runPrediction(input):
    filename = 'finalized_model.sav'
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    pred = loaded_model.predict(input)
    return pred[0]

@app.route('/')
def home():
    return render_template('home2_recoded.html')

@app.route('/model')
def model():
    return render_template('home.html')

@app.route('/join', methods=['GET','POST'])
def my_form_post():
    track = request.form['track']
    artist = request.form['artist']
    result = returnResult(artist, track)
    return jsonify(result=result)

@app.route('/model2', methods=['GET','POST'])
def my_model_post():
    acoust = float(request.form['num1'])/10
    dance = float(request.form['num2'])/10
    duration = float(request.form['num3'])
    energy = float(request.form['num4'])/10
    inst = float(request.form['num5'])/10
    key = float(request.form['num6'])
    live = float(request.form['num7'])/10
    loud = float(request.form['num8'])
    mode = float(request.form['num9'])
    speech = float(request.form['num10'])/10
    tempo = float(request.form['num11'])
    times = float(request.form['num12'])
    valence = float(request.form['num13'])/10
    duration = duration * 1000

    input = [duration, acoust, dance, energy, inst, key, live, loud, mode, speech, tempo, times, valence]
    input = np.asarray(input).reshape(1, -1)
    pred = int(runPrediction(input))
    result = {
        "output": pred
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)