import glob
import os
import re
import pandas as pd

def theme_issues(theme_dir):
    theme_id = os.path.dirname(theme_dir).split('/')[-1]
    image_list = os.listdir(theme_dir)
    issue_list = set()
    for image_path in image_list:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_name = re.sub(r'\d+', '', image_name)
        image_name = re.sub('__','',image_name)
        issue_list.add(image_name)
    return theme_id, list(issue_list)

data = {}
for theme in os.listdir("./resource/pass/"):
    theme_id, issue_list = theme_issues(f"./resource/pass/{theme}/")
    data[theme_id] = issue_list

theme_id, issue_list = theme_issues('./resource/default/')
data[theme_id] = issue_list

max_length = max(len(issues) for issues in data.values())

padded_data = {theme: issues + [None] * (max_length - len(issues)) 
               for theme, issues in data.items()}

df = pd.DataFrame(padded_data)
# print(df)

df_melted = df.melt(var_name = 'theme', value_name = 'issue').dropna()
df_melted['value'] = 1

df_pivot = df_melted.pivot_table(index = ['theme'], columns = ['issue'], values = ['value'], aggfunc = 'sum', fill_value = 0)

issue_sums = df_pivot.sum()

total_theme = len(df_pivot)
common_issues = issue_sums[issue_sums == total_theme]

print(common_issues.index.tolist())



