from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the model, vectorizer, and label encoder
model = joblib.load('health_modell.pkl')
vectorizer = joblib.load('vectoria.pkl')
label_encoder = joblib.load('label_encoderr.pkl')

@app.route('/')
def home():
    return render_template('health.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form['symptoms']
    symptoms_vectorized = vectorizer.transform([symptoms])
    prediction = model.predict(symptoms_vectorized)
    predicted_disease = label_encoder.inverse_transform(prediction)
    return render_template('health.html', prediction_text=f'Predicted Disease: {predicted_disease[0]}')

if __name__ == "__main__":
    app.run(debug=True)
