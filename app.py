from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vectorizer.transform(data).toarray()
        prediction = model.predict(vect)
        result = "Spam" if prediction == 1 else "Not Spam"
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
