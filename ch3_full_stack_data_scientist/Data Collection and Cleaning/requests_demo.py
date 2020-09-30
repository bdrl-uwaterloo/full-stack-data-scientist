import requests
url = "https://en.wikipedia.org/wiki/Waterloo,_Ontario"
page = requests.get(url)
print(page.status_code)