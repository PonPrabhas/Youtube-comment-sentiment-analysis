# Youtube-comment-sentiment-analysis
YouTube comments sentiment analysis

About this project
In this project, I developed a YouTube comment sentiment Web Application,The overall pipeline involved scraping comments from YouTube, followed by data preprocessing and feature extraction using TF-IDF. The Random Forest model was then trained 
on a labelled dataset to classify comments into positive, negative, or neutral sentiments. The 
trained model was integrated into a web application that allows users to input YouTube video 
URLs, scrape the comments, and visualize the sentiment results. This solution provides an 
intuitive and user-friendly way to understand the general sentiment expressed in YouTube 
comment sections

In this project, You first need to run YouTubeTrain.ipynb file, then you got model.pkl and vectorizer.pkl 
Replace the pickle files in your app.py code
and run the flask app

This web application uses Pycharm for app with Flask framework for the web applicaiton and Google colab for training the model
