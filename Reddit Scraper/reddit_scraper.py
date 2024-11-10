import praw
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

#https://www.reddit.com/prefs/apps
class RedditScraper:
  def __init__(self, client_id, client_secret, user_agent):
    self.api = praw.Reddit(
      client_id=client_id,
      client_secret=client_secret,
      user_agent=user_agent)
  def get_comments_from_post(self, post_url):
    submission = self.api.submission(url=post_url)
    
    submission.comments.replace_more(limit=None)
    
    comments_data = []
    for comment in submission.comments.list():
        # Each comment has the user, time, message, and link to its parent (if it's a reply)
        comment_info = {
            "comment_id": comment.id,
            "user": comment.author.name if comment.author else "Deleted",
            "time": datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z'),
            "message": comment.body,
            "parent_id": comment.parent_id.split('_')[1] if comment.parent_id != comment.link_id else None
        }
        comments_data.append(comment_info)
    
    return comments_data
  
  def scrape_user_political_comments(self, username, political_subreddits):
    # Get the Reddit user
    user = self.api.redditor(username)
    
    # List to store comment data
    comments_data = []

    # Print subreddits where the user has commented
    print(set([comment.subreddit.display_name for comment in user.comments.new(limit=None)]))
    
    # Iterate through the user's comments
    for comment in user.comments.new(limit=None):  # `limit=None` fetches all comments
        # Filter by political subreddits
        if comment.subreddit.display_name in political_subreddits:
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