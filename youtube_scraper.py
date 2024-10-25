import googleapiclient.discovery
import pandas as pd
import re

def extract_video_id(youtube_url):
    """
    Extracts the video ID from a YouTube URL.
    Handles both long and short URL formats.
    """
    # Regular expression to handle different YouTube URL formats
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", youtube_url)
    return video_id_match.group(1) if video_id_match else None

def get_video_details(youtube_url):
    """
    Retrieves video details (thumbnail, description, title) from YouTube API.
    """
    api_key = "YOUR API KEY"  # Replace this with your actual YouTube API key
    video_id = extract_video_id(youtube_url)

    if not video_id:
        return None, None, None  # Return None if the video ID cannot be extracted

    # Create a YouTube API client
    youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)

    # Request video details
    request = youtube.videos().list(
        part='snippet',
        id=video_id
    )
    response = request.execute()

    # Extract thumbnail and description
    if response.get('items'):
        snippet = response['items'][0]['snippet']
        thumbnail_url = snippet['thumbnails']['high']['url']
        description = snippet['description']
        title = snippet['title']
        return thumbnail_url, description, title

    return None, None, None

def scrape_comments(youtube_url, max_comments=15):
    """
    Scrapes comments from a YouTube video.
    Limits the number of comments to max_comments.
    """
    api_key = "YOUR API KEY"  # Replace this with your actual YouTube API key
    video_id = extract_video_id(youtube_url)

    if not video_id:
        return None  # Return None if the video ID cannot be extracted

    # Create a YouTube API client
    youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)

    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        # Request comments
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,  # Max comments returned per request
            textFormat='plainText',
            pageToken=next_page_token
        )
        response = request.execute()

        # Process each comment
        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            like_count = item['snippet']['topLevelComment']['snippet']['likeCount']
            comments.append({'comment': comment, 'likes': like_count})

            # Stop if we reach the required number of comments
            if len(comments) >= max_comments:
                break

        # Check if more pages of comments are available
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    # Save comments to CSV (without sorting by likes)
    df = pd.DataFrame(comments)
    csv_file = 'static/comments.csv'
    df.to_csv(csv_file, index=False)

    return csv_file
