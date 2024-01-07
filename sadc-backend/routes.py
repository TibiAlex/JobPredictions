from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics import accuracy_score
import json
from model import JobMatching
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)
jobMatching = JobMatching(3)
# initial_dataset = jobMatching.create_dataset()
initial_dataset = pd.read_csv('./data/dataset.csv')
old_queue_X, old_queue_y = jobMatching.preprocessing_data('init_dataset', False, initial_dataset)

new_queue_X = np.empty((0, 6))
new_queue_y = pd.Series()

@app.route('/trainModel')
def trainModel():
    global new_queue_X
    global new_queue_y
    global old_queue_X
    global old_queue_y

    new_queue_X = np.concatenate([old_queue_X, new_queue_X])
    new_queue_y = pd.concat([old_queue_y, new_queue_y])
    X_train, X_test, y_train, y_test = train_test_split(new_queue_X, new_queue_y, test_size=0.2, random_state=42)
    jobMatching.trainKNN(X_train, y_train)
    y_pred = jobMatching.predictWithKNN(X_test)
    accuracy = accuracy_score(list(y_test), list(y_pred))
    old_queue_X = new_queue_X
    old_queue_y = new_queue_y
    
    new_queue_X = np.empty((0, 6))
    new_queue_y = pd.Series()
    return f'Successfully tarained with accuracy: {accuracy}'


@app.route('/sendData', methods=['POST'])
def sendData():
    global new_queue_X
    global new_queue_y

    data = json.loads(request.data)
    input_df = pd.json_normalize(data)
    [newX, pred] = jobMatching.predictForOnlyOneInput(input_df)
    new_queue_X = np.concatenate([new_queue_X, newX])
    new_queue_y = pd.concat([new_queue_y, pd.Series(pred)])

    indx = old_queue_y[old_queue_y == pred[0]].index[0]

    return jsonify(initial_dataset.loc[initial_dataset.index[indx], 'industries'])
