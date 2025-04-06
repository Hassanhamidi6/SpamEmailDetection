from flask import Flask,render_template,jsonify,request
import joblib
from logger import logging


app=Flask(__name__)

model_path=r"artifacts\data_preprocessing\model.pkl"
vectorizer_path=r"artifacts\data_preprocessing\preprocessor.pkl"

labels=['Not a Spam Email',"Spam Email"]

def Load_Saved_Artifacts(model_path,vectorizer_path):
    vectorizer=joblib.load(vectorizer_path)
    model=joblib.load(model_path)
    return model,vectorizer

@app.route('/')
def main():
    return render_template("index.html")
@app.route('/response',methods=['POST'])
def respond():
    data=request.get_json()
    email=data['email']
    model,vectorizer=Load_Saved_Artifacts(model_path,vectorizer_path)

    email=vectorizer.transform(email)
    prediction=model.predict(email)
    label=labels[prediction[0]]

    return jsonify({"response":label})



if __name__ == "__main__":
    app.run(debug=True)
