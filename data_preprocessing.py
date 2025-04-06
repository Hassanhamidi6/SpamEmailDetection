from pathlib import Path
import os
import pandas as pd
from logger import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
import joblib
import seaborn as sn
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self,
                 dataset_dir:Path,preprocess_data_path:Path,root_dir:Path):
        self.dataset_dir=dataset_dir
        self.preprocess_data_path=preprocess_data_path
        self.root_dir=root_dir
    
    def save_confusion_matrix(self,y_true, predictions, file_path):
        cm = confusion_matrix(y_true, predictions)
        plt.figure(figsize=(10, 7))
        sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                yticklabels=["not spam", "spam"], 
                xticklabels=["not spam", "spam"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        plt.savefig(file_path)
        plt.close()

    def save_metrix(self,y_true,predictions,path):
        accuracy=round(accuracy_score(y_true,predictions)*100)
        precision=round(precision_score(y_true,predictions)*100)
        recall=round(recall_score(y_true,predictions)*100)
        f1=round(f1_score(y_true,predictions)*100)

        plt.figure(figsize=(10,8))
        plt.bar(["accuracy","precision","recall","f1_score"],[accuracy,precision,recall,f1])
        plt.xlabel("Labels")
        plt.ylabel("Values")
        plt.title("Analyzing Different Metrics For Model Performance")
        plt.savefig(path)
        plt.close()
        return accuracy,precision,recall,f1



    def main(self):
        try:

            logging.info(f"Creating Directory at {self.root_dir}")
            os.makedirs(self.root_dir,exist_ok=True)

            logging.info(f"Loading data from {self.dataset_dir}")
            df=pd.read_csv(self.dataset_dir)
            print(df.shape)
            
            logging.info(f"Data Preprocesssing Stage ----> start")
            logging.info("Droping N/A values from dataset")
            df.dropna(inplace=True)
            print(df.shape)
            
            logging.info("Visualizing Some Of data points")
            print(df.sample(5))


            print(df.spam.value_counts())
            logging.info(f"Balancing Data ----> start")
            not_spam_data_points=df[df.spam == 0]
            spam_data_points=df[df.spam == 1]
            higher_value=not_spam_data_points.shape[0]
            spam_data_points=spam_data_points.sample(higher_value,replace=True)
            df=pd.concat([not_spam_data_points,spam_data_points])
            print(df.spam.value_counts())
            logging.info(f"Balancing Data ----> Done")
            logging.info(f"Data Preprocesssing Stage ----> complete")


            logging.info(f"Feature Engineering Stage----> start")
            cv=CountVectorizer()
            X = cv.fit_transform(df['text'])
            Y=df['spam']
            logging.info(f"Feature Engineering Stage----> Complete")

            logging.info(f"Splitting Data into training and test")
            X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

            logging.info(f"Model Training ----> start")
            model=MultinomialNB()
            model.fit(X_train,y_train)
            logging.info(f"Model Training ----> Complete")
           
            
            predictions=model.predict(X_test)
            
            
            metrics_directory=r"artifacts\metrics"
            logging.info(f"Creating Directory at {metrics_directory}")
            os.makedirs(metrics_directory,exist_ok=True)

            c_path=os.path.join(metrics_directory,"confusion_matrix.png")
            logging.info(f"Saving Confusion Matrix at directory {c_path}")
            self.save_confusion_matrix(y_test,predictions,c_path)

            m_path=os.path.join(metrics_directory,"metrix.png")
            accuracy,precision,recall,f1=self.save_metrix(y_test,predictions,m_path)
            logging.info(f"Model Accuracy: {accuracy}\nModel Precision: {precision}\nModel Recall: {recall}\nModel F1: {f1}")

            
            logging.info(f"Saving Model and Vectorizer")
            joblib.dump(cv,"artifacts/data_preprocessing/preprocessor.pkl")
            joblib.dump(model,"artifacts/data_preprocessing/model.pkl")
            
        except Exception as e:
            logging.info(e)