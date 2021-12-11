import pandas as pd

df = pd.read_csv ('prueba.csv')
print(df)

print('valor: ',df.iloc[0].loc['ID'])

import requests

images = []
videos = []
for i in range(5):
    r = requests.get('https://api.jikan.moe/v3/anime/'+str(df.iloc[i].loc['ID']))
    image=r.json()['image_url']
    print(r.json()['image_url'])
    video=r.json()['trailer_url']
    print(r.json()['trailer_url'])
    images.append(image)
    videos.append(video)

df['Image'] = images
df['Trailer'] = videos

print(df)
