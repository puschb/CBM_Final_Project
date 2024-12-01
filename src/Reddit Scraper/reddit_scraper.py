import praw
import prawcore  # Add this import for prawcore exceptions
import csv
import pandas as pd
from datetime import datetime, timezone
import os
import time
from dotenv import load_dotenv
load_dotenv()

class RedditScraper:
    def __init__(self, client_id, client_secret, user_agent):
        self.api = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

    def get_comments_from_post(self, post_id):
        """Retrieve all comments from a specific post by post ID, including those in 'MoreComments'."""
        
        # Retry mechanism for handling 429 Too Many Requests
        while True:
            try:
                submission = self.api.submission(id=post_id)
                submission.comments.replace_more(limit=None)  # Set limit=None to replace all MoreComments objects
                break  # Break out of the loop if no error
            except prawcore.exceptions.TooManyRequests as e:
                print(f"Rate limit hit for comments of post {post_id}. Sleeping for {e.retry_after} seconds.")
                time.sleep(61.0)  # Sleep for the time suggested by the error
                continue  # Retry the request after waiting

        comments_data = []
       
        
        for comment in submission.comments.list():
            # Ensure we are skipping the 'MoreComments' objects and process real comments
            if isinstance(comment, praw.models.MoreComments):
                continue  # We skip MoreComments, as we already replaced them

            comment_info = {
                "comment_id": comment.id,
                "user": comment.author.name if comment.author else "Deleted",
                "time": datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z'),
                "message": comment.body,
                "parent_id": comment.parent_id.split('_')[1] if comment.parent_id != comment.link_id else None
            }
            comments_data.append(comment_info)
        
        return comments_data

    def scrape_subreddit_posts(self, subreddit_name, save_path):
        """Scrape all posts from a specified subreddit with > 100 comments and save posts and comments to CSV files."""
        
        # Ensure the save path directory exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Path for posts CSV file
        posts_csv_file = os.path.join(save_path, "posts.csv")
        
        # Read the existing posts from the CSV (if it exists)
        if os.path.isfile(posts_csv_file):
            existing_posts_df = pd.read_csv(posts_csv_file)
            existing_post_ids = set(existing_posts_df['post_id'])
        else:
            existing_post_ids = set()

        # Check if posts file exists to decide whether to write header
        file_exists = os.path.isfile(posts_csv_file)

        # Retry mechanism for handling 429 Too Many Requests while fetching submissions
        while True:
            try:
                # Iterate over all posts in the specified subreddit
                for submission in self.api.subreddit(subreddit_name).new(limit=None):  # Fetch all available posts
                    if submission.num_comments > 100 and submission.id not in existing_post_ids:
                        # Create a DataFrame for the current post
                        post_data = pd.DataFrame([{
                            "post_id": submission.id,
                            "title": submission.title,
                            "content": submission.selftext,
                            "time": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z'),
                            "num_comments": submission.num_comments
                        }])

                        # Append the post data to the posts CSV using DataFrame to handle special characters
                        post_data.to_csv(
                            posts_csv_file,
                            mode='a',
                            header=not file_exists,  # Write header only if file does not already exist
                            quoting=csv.QUOTE_NONNUMERIC,
                            escapechar='\\',
                            index=False,
                            encoding='utf-8'
                        )

                        # Update `file_exists` to `True` after the first post is appended
                        file_exists = True

                        # Fetch comments for each post and save them to a separate CSV file
                        comments = self.get_comments_from_post(submission.id)
                        comments_df = pd.DataFrame(comments)

                        comments_csv_file = os.path.join(save_path, f"{submission.id}_comments.csv")
                        comments_df.to_csv(
                            comments_csv_file,
                            quoting=csv.QUOTE_NONNUMERIC,
                            escapechar='\\',
                            index=False,
                            encoding='utf-8'
                        )

                        existing_post_ids.add(submission.id)

                        # Print progress message (optional)
                        print(f"Saved post {submission.id} and its comments.")
                break  # Break out of the loop if no error
            except prawcore.exceptions.TooManyRequests as e:
                print(f"Rate limit hit while fetching posts in subreddit {subreddit_name}. Sleeping for {e.retry_after} seconds.")
                time.sleep(61.0)  # Sleep for the time suggested by the error
                continue  # Retry the request after waiting

        print(f"All posts and comments are saved in '{save_path}' directory.")




if __name__ == '__main__':
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    user_agent = os.getenv('USER_AGENT')
    scraper = RedditScraper(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    save_directory = "/home/puschb/UVA/CBM/Information_Spread_Model/Data/Reddit Scrape/"
    scraper.scrape_subreddit_posts(subreddit_name="books", save_path=save_directory)

