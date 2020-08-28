import requests
import os
from time import sleep

downloaded = 0
response = requests.get('http://image-net.org/archive/words.txt')
for index, tag in enumerate(response.text.split('\n')):
    tag = tag.split('\t')
    urls = requests.get('http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}'.format(tag[0])).text
    if 'The synset is not ready yet. Please stay tuned!' not in urls:
        try:
            os.mkdir('./Data/{}'.format(tag[1].replace(' ','_').replace(',','_')))
        except Exception as e:
            print(e)
        print('downloading {}'.format(tag[1]))
        downloaded_photos = 0
        for url_index, url in enumerate(urls.split('\n')):
            try:
                img = requests.get(url)
            except:
                continue

            if 'html' in img.text:
                continue

            file_prefix = None
            if '.png' in url:
                file_prefix = '.png'
            elif '.jpg' in url:
                file_prefix = '.jpg'
            elif '.jpeg' in url:
                file_prefix = '.jpeg'
            elif '.JPG' in url:
                file_prefix = '.JPG'

            if file_prefix:
                with open("./Data/{}/{}{}".format(tag[1].replace(' ','_').replace(',','_'), url_index, file_prefix), "wb") as new_img:
                    new_img.write(img.content)
                    #print("{}{}".format(url_index, file_prefix))
                downloaded_photos += 1


            if downloaded_photos >= 2000:
                break

        downloaded += 1

    if downloaded >= 5000:
        break
    sleep(0.1)
