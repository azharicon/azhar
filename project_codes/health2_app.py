from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the model, vectorizer, and label encoder
model = joblib.load('health_modelll.pkl')
vectorizer = joblib.load('vectoriaa.pkl')
label_encoder = joblib.load('label_encoderrr.pkl')

@app.route('/')
def home():
    return render_template('health2.html')

@app.route('/diagnosis')
def diagnosis():
    return render_template('diagnosis.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form['symptoms']
    symptoms_vectorized = vectorizer.transform([symptoms])
    
    # Predict the disease
    prediction = model.predict(symptoms_vectorized)
    predicted_disease = label_encoder.inverse_transform(prediction)
    
    # Create the prediction text
    prediction_text = f'Predicted Disease: {predicted_disease[0]}'
    
    return render_template('result.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
