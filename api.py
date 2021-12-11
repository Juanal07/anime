import requests

r = requests.get('https://api.jikan.moe/v3/anime/22')
print(r.json()['image_url'])
print(r.json()['trailer_url'])
