import praw
import prawcore  # Add this import for prawcore exceptions
import csv
import pandas as pd
from datetime import datetime, timezone
import os
import time
from dotenv import load_dotenv
import json
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

    def scrape_user_political_comments(self, username, subreddits, limit=None):
        # Get the Reddit user
        user = self.api.redditor(username)

        # List to store comment data
        comments_data = []
        # Print subreddits where the user has commented

        print(set([comment.subreddit.display_name for comment in user.comments.new(limit=limit)]))
        
        # Iterate through the user's comments
        for comment in user.comments.new(limit=limit):  # `limit=None` fetches all comments
            # Filter by political subreddits
            if comment.subreddit.display_name in subreddits:
                # Collect comment data
                comment_info = {
                    "comment_id": comment.id,
                    "subreddit": comment.subreddit.display_name,
                    "body": comment.body,
                    "time": datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z'),
                    "score": comment.score,
                    "link_id": comment.link_id,  # ID of the post this comment is related to
                    "parent_id": comment.parent_id  # ID of the comment or post this comment replies to
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

class ArcticShiftScraper:
    def process_comments(input_path, output_path):
        field_mappings = {
            "comment_id": "id",
            "parent_comment_id": "parent_id",
            "time_stamp_created": "created_utc",
            "comment_text": "body",
            "user": "author",
            "corresponding_post_id": "link_id",
            "link" : "permalink"
        }

        data = []

        # Read the JSON Lines input file line by line and load valid data into a list
        with open(input_path, 'r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                try:
                    comment = json.loads(line.strip())  
                    if comment["author"] != "[deleted]":  # Filter deleted authors
                        parent_comment_id = comment[field_mappings["parent_comment_id"]][3:]
                        if parent_comment_id == comment[field_mappings["corresponding_post_id"]][3:]:
                            parent_comment_id = "None"
                        data.append({
                            "comment_id": comment[field_mappings["comment_id"]],
                            "parent_comment_id": parent_comment_id,
                            "time_stamp_created": comment[field_mappings["time_stamp_created"]],
                            "comment_text": comment[field_mappings["comment_text"]],
                            "user": comment[field_mappings["user"]],
                            "corresponding_post_id": comment[field_mappings["corresponding_post_id"]][3:],
                            "link": comment[field_mappings["link"]]
                        })
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")

        comments_df = pd.DataFrame(data)
        comments_df.to_csv(
            output_path,
            quoting=csv.QUOTE_NONNUMERIC,
            escapechar='\\',
            index=False,
            encoding='utf-8'
        )

        print(f"Comments successfully written to {output_path}")
    def process_posts(input_path, output_path):
        field_mappings = {
            "post_id": "id",
            "title": "title",
            "content": "selftext",
            "timestamp": "created_utc",
            "num_comments": "num_comments",
            "link": "url"
        }

        data = []

        # Read the JSON Lines input file line by line and load valid data into a list
        with open(input_path, 'r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                try:
                    post = json.loads(line.strip())
                    if post[field_mappings["num_comments"]] < 100:
                        continue
                    data.append({
                        "post_id": post[field_mappings["post_id"]],
                        "title": post[field_mappings["title"]],
                        "content": post[field_mappings["content"]],
                        "timestamp": post[field_mappings["timestamp"]],
                        "num_comments": post[field_mappings["num_comments"]],
                        "link": post[field_mappings["link"]]
                    })
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")

        posts_df = pd.DataFrame(data)

        posts_df.to_csv(
            output_path,
            quoting=csv.QUOTE_NONNUMERIC,
            escapechar='\\',
            index=False,
            encoding='utf-8'
        )

        print(f"Post data successfully written to {output_path}")


if __name__ == '__main__':
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    user_agent = os.getenv('USER_AGENT')
    scraper = RedditScraper(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    save_directory = "/home/puschb/UVA/CBM/Information_Spread_Model/Data/Reddit Scrape/"
    scraper.scrape_subreddit_posts(subreddit_name="books", save_path=save_directory)

