import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from collections import Counter
import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
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

def load_data(messages_filepath, categories_filepath):
    '''
    input (
        messages_filepath: csv
        categories_filepath: csv
        )
    output (
        df : pandas dataframe object
        )
    
    This function loads/extracts the data from messages_filepath,categories_filepath csv files and returns the merged dataframe.
    '''
    ###Extract###
    #Load data from files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #Merge data on id
    df = messages.merge(categories,on='id')
    
    return df
        
def clean_data(df):
    '''
    input (
        df : pandas dataframe object
        )
    output (
        df : pandas dataframe object
        )
    
    This function cleans the data and structures it in a proper format to be used for machine learning models.
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x : x.split('-')[0]).values
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x.split('-')[1] if len(x.split('-')) >= 2 else x )
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df.drop(['categories'],axis=1,inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df
    


def save_data(df, database_filename):
    '''
    input (
        df: pandas dataframe object
        database_filename: str object
        )
    
    This function loads the data obtained after the cleaning process into sql database and saves some analyzed data for front end               visualizations.
    '''
    ###Load###
    #Load the data into sql table
    file_name = 'sqlite:///'+database_filename
    engine = create_engine(file_name)
    df.to_sql('disasterResponseData', engine, index=False)  
    tokenized_message=df.message.apply(lambda x : tokenize(x))
    counter=Counter()
    for message in tokenized_message.values:
        for token in message:
            if len(token)>2 :
                counter[token] += 1
    # top 10 words 
    top = counter.most_common(10)
    top_words = [i[0] for i in top]
    top_counts = [i[1] for i in top]
    np.savez('data/top_10_word_count.npz', top_words=top_words, top_counts=top_counts)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()