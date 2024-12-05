import numpy as np
import pandas as pd
import regex as re
import os
import csv
import glob

def clean_comments(path_to_csv, save_path, prune = True):
    df = pd.read_csv(path_to_csv, quoting=csv.QUOTE_NONNUMERIC,
            escapechar='\\',
            encoding='utf-8')
    print(f'Total Comments (uncleaned): {len(df)}')

    df['comment_text'] = df['comment_text'].astype(str)

    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub(r'https?:\/\/\S+', '', x)) # remove links
    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x)) #remove links

    # leave emojis in for now since tokenizer can handle it
    #df['comment_text'] = df['comment_text'].apply(lambda x: re.sub(EMOJI_PATTERN,'',x)) #remove emojis

    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub(r"(\x5Co\x2F)",'',x)) #remove \o/ emoji
    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub(r"(\x5C_\x28\x29_\x2F)",'',x)) #remove \_()_/ emoji

    # for improper html alphanumerical conversions
    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub(r"(&amp;#8211;)", '–', x)) #endash
    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub(r"(&amp;)", '&', x))  # & symbol
    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub(r"(&lt;)", '<', x))  # less than symbol
    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub(r"(&gt;)", '>', x))  # greater than symbol
    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub(r"(&#39;)", "'", x))  # single quote
    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub(r"(&quot;)", '"', x))  # double quote

    #deal with special characters
    #df['comment_text'] = df['comment_text'].apply(lambda x: re.sub(r"([\+\*])\1+",'',x))#only remove + and * when they are repeated for visual attention
    #df['comment_text'] = df['comment_text'].apply(lambda x: re.sub(r"(?<=!)!|(?<=\()\(|(?<=\))\)|(?<=-)-|(?<=\?)\?|(?<=\|)\|", '', x)) #remove repeating special characters

    #df['comment_text'] = df['comment_text'].str.lower() #make lowercase --> keep casing for now

    #remove mentions
    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub(r"@\x20[a-zA-Z0-9_]*|@[a-zA-Z0-9_]*",'',x))


    # remove extra enters and spaces
    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub(r"<br>", '\x20', x))
    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub(r"\x20+", '\x20', re.sub(r"\u000a+|\u000d+|\u2028+|\u2029+", '\x20', x)))

    if prune:
        print(f'Total Comments Before Pruning: {len(df)}')
        #prune out comments that are not part of a branch of length 4 or greater
        df = filter_comments_by_branch_length(df, min_branch_length= 2)
        print(f'Total Comments After Pruning: {len(df)}')
    
    print(f'Total Cleaned Comments: {len(df)}')

    df.to_csv(save_path, quoting=csv.QUOTE_NONNUMERIC,
            escapechar='\\',
            index=False,
            encoding='utf-8')

def filter_comments_by_branch_length(df, min_branch_length=2):
    # Filters comments where the entire branch (from root to leaf) has fewer than `min_branch_length` comments.

    parent_to_children = {}
    for parent, child in zip(df['parent_comment_id'], df['comment_id']):
        if parent not in parent_to_children:
            parent_to_children[parent] = []
        parent_to_children[parent].append(child)

    def get_deepest_leaf(comment):
        if comment[0] not in parent_to_children:
            return comment
        return max([get_deepest_leaf((child, comment[1] + 1)) for child in parent_to_children[comment[0]]], key=lambda x: x[1])

    def calculate_branch_length(comment_id):
        branch_length = 0
        while True:
            result = df.loc[df['comment_id'] == comment_id, 'parent_comment_id']
            parent_id = result.values[0] if not result.empty else None
            if parent_id is None:
                break
            else:
                branch_length += 1
                comment_id = parent_id
        return branch_length
    
    comment_to_branch_length = {}
    for idx, comment in enumerate(df['comment_id']):
        if idx % 1000 == 0:
            print(idx)
        deepest_leaf = get_deepest_leaf((comment, 0))[0]
        branch_length = calculate_branch_length(deepest_leaf)
        comment_to_branch_length[comment] = branch_length

    print(comment_to_branch_length)
    

    df['branch_length'] = df['comment_id'].map(comment_to_branch_length)
    df_filtered = df[df['branch_length'] >= min_branch_length].drop(columns=['branch_length'])

    return df_filtered


if __name__ == '__main__':
    print('starting')    
    df = pd.DataFrame({
        'comment_id': ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'],
        'parent_comment_id': ['None', 'c1', 'None', 'c2', 'c2', 'c5', 'c6', 'c6']
    })

    df = filter_comments_by_branch_length(df,min_branch_length=4)
    print(df)
