import glob
import os
import re

theme_list = os.listdir("./resource/pass")
issue_list = set()
for theme in theme_list:
    image_list = glob.glob(f"./resource/pass/{theme}/*.png")
    for image in image_list:
        image_name = os.path.basename(image)
        image_name = re.sub(r'\d+', '', image_name)
        issue_list.add(image_name)
print(issue_list)


