import requests
import json

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'}
response = requests.get("https://reddit.com/api/trending_subreddits.json", headers=headers)
data = response.json()
subreddit = data["subreddit_names"]
print(json.dumps(data, indent=4))
print(subreddit)