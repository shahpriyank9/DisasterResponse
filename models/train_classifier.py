import sys
import pandas as pd
import numpy as np
import re
import sqlalchemy
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    input: (
        database_filepath: path to database : str
    )
    output: (
        X: features : dataframe 
        y: target : dataframe
        category_names: names of targets : series
    )
    This function loads the data from database file and returns features,target and target_names
    '''
    database_name = 'sqlite:///'+database_filepath
    engine = create_engine(database_name)
    df = pd.read_sql_table('disasterResponseData',engine)
    X = df.message.values
    Y = df.drop(['id', 'message', 'original', 'genre'],axis=1)
    categories=Y.columns
    return  X,Y,categories


def tokenize(text):
    '''
    input: (
    text: Message to tokenize : str
    )
    output: (
    clean_tokens : list of cleaned vectorized tokens : list
    )
    This function is used to get cleaned vecotrized tokens from meassage to be classified
    '''
    #Normalizing the sentence(Removed punctuation , converted to lower case)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    #tokenize the message
    tokens = word_tokenize(text)
    #Remove stop words
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    #lemmatize 
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    output(
       cv : outputs model to be trained using grid search cv : object
    )
    This function instanciates the model using Pipeline and grid search cv and returns the object
    '''
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ]) 
    
    param_grid = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__min_samples_split': [5, 10],
        #'clf__estimator__min_samples_leaf':[2,5]
        #'clf__estimator__n_estimators': [100, 250]
    }
    
    cv = GridSearchCV(pipeline, param_grid=param_grid, verbose=2)
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    input(
        model: model to be evaluated : object
        X_test: testing features : dataframe
        Y_test: testing target : dataframe
        category_names: list of clases : series
    )
    This function prints the evalutaion of the model with metrics such as F1 score, precision , recall and overall accuracy against the         testing data.
    '''
    #predictions 
    y_pred = model.predict(X_test)
    #classification report
    for i,col in enumerate(category_names.values):
        print("Evaluation Values for column '{}'".format(col))
        print(classification_report(Y_test[col].values,y_pred[:,i]))
    #accuracy
    print('Accuracy : {}'.format(np.mean(Y_test.values == y_pred)))



def save_model(model, model_filepath):
    '''
    input(
        model: model to be saved : object
        model_filepath: path where model is to be saved : str
    )
    This function saves the model in a pickle file at the path specified
    '''
    #Save the model in pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()