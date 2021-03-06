import json
import plotly
import pandas as pd
import numpy as np
import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disasterResponseData', engine)
# load model
model = joblib.load("../models/classifier.pkl")
#load visualization
top_10_word_count = np.load('../data/top_10_word_count.npz')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #Target count percentage
    class_percentage = df.drop(['id','message','original','genre','related'], axis=1).mean()*100
    class_names = list(class_percentage.index)
    
    #Top 10 word count
    top_words=list(top_10_word_count['top_words'])
    top_counts=list(top_10_word_count['top_counts'])
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data':[
                Bar(
                    x=class_names,
                    y=class_percentage
                )
            ],
            'layout':{
                'title':'Percentage of data per class',
                'xaxis': {
                    'title': 'Class'
                },
                'yaxis': {
                    'title': 'Percentage'
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_words,
                    y=top_counts
                )
            ],

            'layout': {
                'title': 'Top 10 word count',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()