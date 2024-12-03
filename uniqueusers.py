import os
import pandas as pd

def get_unique_users(folder_path):

    csv_files = [file for file in os.listdir(folder_path) if file.endswith('_comments.csv')]
    
    unique_users = set()
    
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        try:
            df = pd.read_csv(file_path, usecols=['user'])
            unique_users.update(df['user'].dropna().unique())
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    unique_users_list = sorted(unique_users)
    return unique_users_list

folder_path = "Data/Reddit Scrape"

unique_users = get_unique_users(folder_path)

output_file = os.path.join(folder_path, "unique_users.csv")
pd.DataFrame(unique_users, columns=["user"]).to_csv(output_file, index=False)