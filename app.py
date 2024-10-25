from flask import Flask, request, render_template, send_file, redirect, url_for
from youtube_scraper import scrape_comments, get_video_details
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import pickle
from scipy.sparse import issparse  # Import to check for sparse matrices

app = Flask(__name__)

# Path to save and serve files
STATIC_DIR = 'static'
COMMENTS_FILE = os.path.join(STATIC_DIR, 'comments.csv')
CHART_FILE = 'sentiment_pie_chart.png'  # Pie chart filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    url = request.form['url']

    # Get video details
    thumbnail_url, description, title = get_video_details(url)

    if not thumbnail_url or not description or not title:
        return render_template('index.html', error="Invalid YouTube URL or Video not found.")

    # Scrape comments and save to CSV (scraping up to 100 comments)
    csv_file = scrape_comments(url, max_comments=100)

    # Redirect to sentiment analysis after scraping
    if csv_file:
        return redirect(url_for('analyze', thumbnail_url=thumbnail_url, description=description, title=title))
    else:
        return render_template('index.html', error="Failed to scrape comments")

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if not os.path.exists(COMMENTS_FILE):
        return redirect(url_for('index'))

    # Get video details from request args
    thumbnail_url = request.args.get('thumbnail_url')
    description = request.args.get('description')
    title = request.args.get('title')

    # Load the pre-trained model and vectorizer
    with open('model.pkl', 'rb') as model_file, open('vectorizer.pkl', 'rb') as vectorizer_file:
        model = pickle.load(model_file)
        vectorizer = pickle.load(vectorizer_file)

    # Load the comments CSV
    df = pd.read_csv(COMMENTS_FILE)

    # Transform comments using the vectorizer
    comments = df['comment'].tolist()
    comments_transformed = vectorizer.transform(comments)

    # Predict sentiment using the loaded model
    sentiment_predictions = model.predict(comments_transformed)

    # Map predictions to sentiment labels
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    df['sentiment'] = [sentiment_map.get(pred, 'unknown') for pred in sentiment_predictions]

    # Save the updated DataFrame with sentiment
    df_result = df[['comment', 'sentiment']]
    df_result.to_csv(COMMENTS_FILE, index=False)

    # Calculate the sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()

    # Plot pie chart for sentiment distribution
    plt.figure(figsize=(6, 6))
    sentiment_counts.plot.pie(autopct='%1.1f%%', colors=['red', 'blue', 'green'], labels=sentiment_counts.index)
    plt.title('Sentiment Analysis Distribution')
    plt.ylabel('')  # Hide y-label for better aesthetics

    # Ensure STATIC_DIR exists
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR)

    # Save pie chart to the static directory
    chart_file_path = os.path.join(STATIC_DIR, CHART_FILE)
    plt.savefig(chart_file_path)
    plt.close()

    # Count positive, neutral, and negative comments
    positive_count = sentiment_counts.get('positive', 0)
    neutral_count = sentiment_counts.get('neutral', 0)
    negative_count = sentiment_counts.get('negative', 0)

    # Render the updated template with video details and counts
    return render_template(
        'index.html',
        analysis=True,
        csv_file='comments.csv',
        chart_file=CHART_FILE,
        comments=df_result.to_dict(orient='records'),
        thumbnail_url=thumbnail_url,
        description=description,
        title=title,
        positive_count=positive_count,
        neutral_count=neutral_count,
        negative_count=negative_count
    )

@app.route('/details')
def details():
    if not os.path.exists(COMMENTS_FILE):
        return redirect(url_for('index'))

    # Load the comments CSV
    df = pd.read_csv(COMMENTS_FILE)

    # Load the pre-trained model and vectorizer
    with open('model.pkl', 'rb') as model_file, open('vectorizer.pkl', 'rb') as vectorizer_file:
        model = pickle.load(model_file)
        vectorizer = pickle.load(vectorizer_file)

    # Transform comments using the vectorizer
    comments = df['comment'].tolist()
    comments_transformed = vectorizer.transform(comments)

    # Predict sentiment using the loaded model
    sentiment_predictions = model.predict(comments_transformed)

    # Map predictions to sentiment labels
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    df['sentiment'] = [sentiment_map.get(pred, 'unknown') for pred in sentiment_predictions]

    # Prepare data for detailed view
    detailed_data = []
    for comment, vector, pred in zip(df['comment'], comments_transformed, sentiment_predictions):
        if issparse(vector):
            vector_list = vector.toarray().tolist()[0]
        else:
            vector_list = vector.tolist()
        detailed_data.append({
            'comment': comment,
            'vectorized_comment': vector_list,  # Convert to list for JSON serialization
            'sentiment': sentiment_map.get(pred, 'unknown')
        })

    return render_template('details.html', detailed_data=detailed_data)

@app.route('/download')
def download():
    if os.path.exists(COMMENTS_FILE):
        return send_file(COMMENTS_FILE, as_attachment=True)
    return redirect(url_for('index'))

@app.route('/process')
def process():
    try:
        if not os.path.exists(COMMENTS_FILE):
            return redirect(url_for('index'))

        # Load the comments CSV
        df = pd.read_csv(COMMENTS_FILE)

        # Load the pre-trained model and vectorizer
        with open('model.pkl', 'rb') as model_file, open('vectorizer.pkl', 'rb') as vectorizer_file:
            model = pickle.load(model_file)
            vectorizer = joblib.load(vectorizer_file)

        # Original comments
        comments = df['comment'].tolist()

        # Preprocess the comments using the vectorizer (same process as training)
        comments_transformed = vectorizer.transform(comments)

        # Predict sentiment using the loaded model
        sentiment_predictions = model.predict(comments_transformed)

        # Map predictions to sentiment labels
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

        # Get the feature names (words) from the vectorizer
        feature_names = vectorizer.get_feature_names_out()

        # For each comment, get the top words contributing to the sentiment prediction
        important_words = []
        for i, comment_vector in enumerate(comments_transformed):
            if issparse(comment_vector):
                comment_vector = comment_vector.toarray()

            feature_importance = np.argsort(comment_vector[0])[-3:]  # Top 3 contributing features
            top_words = [feature_names[j] for j in feature_importance]
            important_words.append(", ".join(top_words))

        def preprocess_for_display(comment):
            # Check if the comment is a string
            if not isinstance(comment, str):
                print(f"Invalid comment type: {type(comment)}")  # Debug print
                return comment  # Return as-is if not a string

            try:
                print(f"Original comment: {comment}")  # Debug print
                preprocessed = vectorizer.build_analyzer()(comment)  # Preprocess the comment
                preprocessed_comment = ' '.join(preprocessed)  # Join preprocessed words
                print(f"Preprocessed comment: {preprocessed_comment}")  # Debug print
                return preprocessed_comment  # Return the preprocessed comment
            except Exception as e:
                print(f"Error preprocessing comment '{comment}': {e}")  # Debug print
                return comment  # Return the original comment in case of error

        # Preprocess comments for display
        preprocessed_comments = [preprocess_for_display(comment) for comment in comments]

        # Prepare data for detailed view
        detailed_data = []
        for comment, preprocessed, pred, words in zip(comments, preprocessed_comments, sentiment_predictions, important_words):
            detailed_data.append({
                'comment': comment,
                'preprocessed_comment': preprocessed,
                'sentiment': sentiment_map.get(pred, 'unknown'),
                'important_words': words
            })

        # Render the process template
        return render_template('process.html', detailed_data=detailed_data)

    except Exception as e:
        return str(e)  # Return the error message for debugging

if __name__ == '__main__':
    app.run(debug=True)
