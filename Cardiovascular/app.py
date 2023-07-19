from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the data and preprocess
df = pd.read_csv('Heart_Disease_Prediction.csv')
df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
sc = StandardScaler()
X = sc.fit_transform(X)

# Train the model
forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=1)
forest.fit(X, Y)

# Route for the home page
@app.route('/')
def home():
    return render_template('UI.html')

# Route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the form data
    age = int(request.form['age'])
    sex = int(request.form['Sex'])
    pain = int(request.form['pain'])
    bp = int(request.form['bp'])
    chol = int(request.form['chol'])
    fbs = int(request.form['FBS'])
    ekg = int(request.form['EKG'])
    hr = int(request.form['HR'])
    agina = int(request.form['Agina'])
    st = int(request.form['ST'])
    slope = int(request.form['Slope'])
    vessels = int(request.form['Vessels'])
    thalium = int(request.form['Thalium'])

    # Create a feature vector
    features = np.array([[age, sex, pain, bp, chol, fbs, ekg, hr, agina, st, slope, vessels, thalium]])
    print(features)
    # Scale the features
    features = sc.transform(features)

    # Perform the prediction
    prediction = forest.predict(features)

    # Convert the prediction to a human-readable string
    result = 'Heart Disease: Risk' if prediction == 1 else 'Heart Disease: No'

    return render_template('result.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
