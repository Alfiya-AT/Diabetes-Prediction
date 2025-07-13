from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def hello():
    return render_template('index.html')

# prediction function   
@app.route('/predict', methods = ['POST']) 
def predict(): 
    form_data = request.form

    A = [
    float(form_data['Pregnancies']),
    float(form_data['Glucose']),
    float(form_data['Blood Pressure']),
    float(form_data['Skin Thickness']),
    float(form_data['Insulin']),
    float(form_data['BMI']),
    float(form_data['DiabetesPedigreeFunction']),
    float(form_data['Age'])
]

    # A = [float(x) for x in request.form.values()]
    model_probability = model.predict_proba([A])
    prediction = "Probability of  this user having Diabetes is %0.2f"%model_probability[0][1]
    return render_template('index.html', result = prediction)
if __name__ == "__main__":
    app.run(debug=True)
