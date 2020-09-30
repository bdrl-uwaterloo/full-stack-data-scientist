import requests
import json

headers = {
  'Authorization': 'OAuth oauth_consumer_key="f3KD5zvTAGVIILk7o65dr5PZk",oauth_token="200019274-c2HdP36j7FJTDgxr7S3NayUFfgYlqLbcUirCZ5T4",oauth_signature_method="HMAC-SHA1",oauth_timestamp="1600732571",oauth_nonce="faa4HDPiOMn",oauth_version="1.0",oauth_signature="SxqbJaAs8LAFm3HR%2FRSGQHV9UHk%3D"',
  'Cookie': 'personalization_id="v1_JjQXxRqvI2qq0olSMb/rqA=="; guest_id=v1%3A158951098243717623; lang=en'
}

response = requests.get("https://api.twitter.com/1.1/trends/place.json?id=1", headers=headers)
data = response.json()
trends = data[0]["trends"]
names = [item["name"] for item in trends]
for name in names:
  print(name)