# Disaster Response Pipeline Project
Udacity Data Science Nanodegree project used to display Data Engineering skills like ETL and ML pipeline,using a pretrained model in webapp, display some data visualizations in the Dashboard. This project is built on Python 3 and flask is used for building web app.

This project reads in a message and classifies it as one or more of the 36 pre-defined categories or classes as per the data set on which the model was trained.

### Files:
1. data/process_data.py : Use ETL to  process the data and load it in database file
2. model/train_classifier.py : Train the model with the processed data and save it in a pickle file to be used in the webapp.
3. app/run.py : Initialize the server and the visualizations
4. app/templates : templated used to render html
5. data/* .csv : data used in the project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
