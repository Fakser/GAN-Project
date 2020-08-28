from src.controller import *

def data_load(IMG_SHAPE = IMG_SHAPE, MAX_DATASET_SIZE = MAX_DATASET_SIZE, data_dimensions = 3, path = '.\\Data\\anime_faces\\'):
    data = []
    i = 0
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            #print(os.path.join(dirname, filename))
            img = np.array([0])
            try:
                img = np.asarray(Image.open(os.path.join(dirname, filename)).resize(IMG_SHAPE))
            except Exception as e:
                print(e)
                continue

            if img.shape == (IMG_SHAPE[0], IMG_SHAPE[1], data_dimensions):
                data.append(img)
                print(str(i/MAX_DATASET_SIZE) + '%', end = "\r")
                i+=1

            if len(data) >= MAX_DATASET_SIZE:
                break

    data = np.array(data)
    data = (data  - 127.5) / 127.5
    return data