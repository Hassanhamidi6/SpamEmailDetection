import joblib

cv=joblib.load(r"artifacts\data_preprocessing\preprocessor.pkl")
model=joblib.load(r"artifacts\data_preprocessing\model.pkl")
classes=['not spam','spam']
query=input("Enter the query: \n")
query=cv.transform([query])

prediction=model.predict(query)

print(f"This email is {classes[prediction[0]]}")