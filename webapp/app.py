from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Define the absolute file path to the model
model_path = f"D:\\Innomatics_Internship_tasks\\ML_Task_SentimentAnalysis_Filpkart\\SentimentAnalysis_Flipkart_Review\\webapp\\Best_Model\\naive_bayes.pkl"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        if not review:
            message = "Please enter a review."
            return render_template('index.html', message=message)
        else:
            # Check if the model file exists
            if not os.path.exists(model_path):
                return "Error: Model file not found."
            
            # Load the model
            model = joblib.load(model_path)
            prediction = model.predict([review])[0]
            
            # Map prediction to sentiment
            sentiment = "Positive" if prediction == 1 else "Negative"

            return render_template('output.html', prediction=sentiment)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
