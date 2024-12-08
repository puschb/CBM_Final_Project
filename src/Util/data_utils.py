import csv
import json

def load_reddit_posts(file_path):
    posts = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            posts.append({
                "post_id": row["post_id"],
                "title": row["title"],
                "content": row["content"],
            })
    return posts

def get_comments_for_post(file_path, post_id):
    comments = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["corresponding_post_id"] == post_id:
                comments.append({
                    "comment_id": row["comment_id"],
                    "parent_comment_id": row["parent_comment_id"],
                    "timestamp": float(row["time_stamp_created"]),
                    "comment_text": row["comment_text"],
                    "user": row["user"],
                })
    return comments

def get_user_comment_history(file_path, user, timestamp, post_id):
    comment_history = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["user"] == user and row["corresponding_post_id"]!=post_id and float(row["time_stamp_created"]) > timestamp:
                comment_history.append({
                    "comment_text": row["comment_text"],
                })
    return comment_history

def get_comments_in_post_thread(file_path, comment_id):
    # dict to hold comments by ID for quick access
    comments_map = {}
    thread_comments = []

    # Load the comments into a dictionary for fast lookup ... should be doing this for all the data, but will worry abt this later
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            comments_map[row["comment_id"]] = {
                "comment_id": row["comment_id"],
                "parent_comment_id": row["parent_comment_id"],
                "timestamp": float(row["time_stamp_created"]),
                "comment_text": row["comment_text"],
                "user": row["user"],
                "corresponding_post_id": row["corresponding_post_id"],
                "link": row["link"]
            }

    # traverse comment chain using parent_comment_id
    current_comment_id = comment_id
    while current_comment_id:
        current_comment = comments_map.get(current_comment_id)
        if current_comment:
            thread_comments.append(current_comment)
            current_comment_id = current_comment["parent_comment_id"]
        else:
            break

    return thread_comments

def upload_json_to_file(
    post_title, 
    post_content, 
    previous_comments_on_thread, 
    comment_of_interest,
    user_comment_history, 
    comment_id
    ):
    file_path = f'/home/ujx4ab/ondemand/CBM_Final_Project/Data/post_prompts/{post_title}.json'
    data = {
        "input": {
            "context": {
                "post_title": post_title,
                "post_content": post_content,
                "previous_comments_on_thread": previous_comments_on_thread,
                "comment_of_interest": comment_of_interest,
                "user_comment_history": user_comment_history
            },
            "comment_id": comment_id
        }
    }

    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)