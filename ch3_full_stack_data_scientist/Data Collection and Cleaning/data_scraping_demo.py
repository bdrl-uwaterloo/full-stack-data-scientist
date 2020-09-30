import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/Waterloo,_Ontario"
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
table = soup.find("table", class_ = "wikitable collapsible")
tr = table.find_all("tr")[1:15]
header = [th.get_text().strip('\n') for th in tr[0].find_all("th")]
data = []
for r in tr[1:14]:
    data.append([])
    data[-1].append(r.find("th").get_text().strip('\n'))
    td = [value.get_text().strip('\n') for value in r.find_all("td")]
    data[-1] += td
data.insert(0, header)
for item in data:
    print(item)