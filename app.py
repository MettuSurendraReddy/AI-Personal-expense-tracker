from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    category = ''
    if request.method == 'POST':
        description = request.form['description']
        vectorized_desc = vectorizer.transform([description])
        category = model.predict(vectorized_desc)[0]
    return render_template('index.html', category=category)

if __name__ == '__main__':
    app.run(debug=True)
