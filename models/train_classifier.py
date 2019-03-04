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
    database_name = 'sqlite:///'+database_filepath+'.db'
    engine = create_engine(database_name)
    df = pd.read_sql_table('disasterResponseData',engine)
    X = df.message.values
    Y = df.drop(['id', 'message', 'original', 'genre'],axis=1)
    categories=Y.columns.values
    return  X,Y,categories


def tokenize(text):
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
    
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ]) 
    
    param_grid = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__min_samples_split': [5, 10],
        'clf__min_samples_leaf':[2,5]
        'clf__n_estimators': [100, 250]
    }
    
    cv = GridSearchCV(pipeline, param_grid=param_grid, verbose=2, n_jobs=-1)
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    #predictions 
    y_preds = model.predict(X_test)
    #classification report
    print(classification_report(y_preds, Y_test.values, target_names=category_names))
    #accuracy
    print('Accuracy : {}'.format(np.mean(Y_test.values == y_preds)))



def save_model(model, model_filepath):
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