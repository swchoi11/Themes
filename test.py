from src.issue.icon import Icon
import glob
from src.gemini import Gemini

testimages = glob.glob('./icontestimages/*.png')

for image in testimages:
    icon = Icon(image)
    issues = icon.run_icon_check()
    for issue in issues:
        gemini = Gemini(image)
        issue = gemini.issue_score(issue)
        with open('icontestimages/result-score.txt', 'a') as f:
            f.write(f"{issue.model_dump_json()}\n")
