# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 2025
@author: KOWSIK.S
"""
import os
import flask
from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set working directory
try:
    os.chdir(r"C:\Users\KOWSIK.S\Data_Science\ML\Capstone")
except Exception as e:
    print(f"Error setting directory: {e}")
    exit(1)

# Load shared preprocessors
try:
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/label_encoder_stage_fear.pkl', 'rb') as f:
        le_stage_fear = pickle.load(f)
    with open('models/label_encoder_drained.pkl', 'rb') as f:
        le_drained = pickle.load(f)
except Exception as e:
    print(f"Error loading preprocessor files: {e}")
    exit(1)

app = flask.Flask(__name__, template_folder="templates")

@app.route('/', methods=['GET', 'POST'])
def main():
    try:
        if flask.request.method == "GET":
            return flask.render_template("index.html")

        if flask.request.method == "POST":
            # Collect input from form
            time_spent_alone = float(flask.request.form['time_spent_alone'])
            stage_fear = flask.request.form['stage_fear']
            social_event_attendance = float(flask.request.form['social_event_attendance'])
            going_outside = float(flask.request.form['going_outside'])
            drained_after_socializing = flask.request.form['drained_after_socializing']
            friends_circle_size = float(flask.request.form['friends_circle_size'])
            post_frequency = float(flask.request.form['post_frequency'])
            model_choice = request.form.get('model_choice')

            # Choose the correct model file
            if model_choice == 'logistic':
                filename = 'model_logistic.pkl'
            elif model_choice == 'knn':
                filename = 'model_knn.pkl'
            elif model_choice == 'svm':
                filename = 'model_svm.pkl'
            elif model_choice == 'dt':
                filename = 'model_dt.pkl'
            elif model_choice == 'xgb':
                filename = 'model_xgb.pkl'
            elif model_choice == 'ada':
                filename = 'model_ada.pkl'
            else:
                filename = 'model_rf.pkl'

            with open(f'models/{filename}', 'rb') as f:
                model = pickle.load(f)

            # Build input DataFrame
            input_variables = pd.DataFrame([[time_spent_alone, stage_fear, social_event_attendance,
                                            going_outside, drained_after_socializing, friends_circle_size,
                                            post_frequency]],
                                           columns=['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                                                    'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
                                                    'Post_frequency'],
                                           index=['Input'])

            # Preprocess input
            input_variables['Stage_fear'] = le_stage_fear.transform(input_variables['Stage_fear'])
            input_variables['Drained_after_socializing'] = le_drained.transform(input_variables['Drained_after_socializing'])
            numeric_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                               'Friends_circle_size', 'Post_frequency']
            input_variables[numeric_columns] = scaler.transform(input_variables[numeric_columns])

            # Predict
            prediction = model.predict(input_variables)[0]
            result = 'Introvert' if prediction == 1 else 'Extrovert'

            return flask.render_template('index.html',
                                         original_input={
                                             'Time Spent Alone': time_spent_alone,
                                             'Stage Fear': stage_fear,
                                             'Social Event Attendance': social_event_attendance,
                                             'Going Outside': going_outside,
                                             'Drained After Socializing': drained_after_socializing,
                                             'Friends Circle Size': friends_circle_size,
                                             'Post Frequency': post_frequency,
                                             'Model Used': model_choice
                                         },
                                         result=result)
    except Exception as e:
        return f"An error occurred: {str(e)}"

#if __name__ == '__main__':
#    app.run(debug=False, use_reloader=False)
    

'''🧑‍🍳 What is Waitress?
Waitress is a production-grade WSGI server for Python web applications — like your Flask app.

It’s like a “delivery system” that knows how to serve your app efficiently and safely when users send requests (from browsers, APIs, etc.)

🧩 What WSGI Means
WSGI = Web Server Gateway Interface
It’s a standard that allows Python web frameworks (Flask, Django, FastAPI) to talk to real web servers (Nginx, Apache, etc.)

✅ Why You Chose Waitress
You're on Windows

It’s simple to set up

It removes this WARNING: This is a development server. Do not use it in a production deployment.'''

# Run with waitress (no Flask warning)
from waitress import serve

if __name__ == '__main__':
    serve(app, host='127.0.0.1', port=5000)
    print('Project running on http://127.0.0.1:5000/')
    

# Project running on http://127.0.0.1:5000/