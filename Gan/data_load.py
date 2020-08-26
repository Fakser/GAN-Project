from Gan.controller import *
def data_load(IMG_SHAPE = IMG_SHAPE, MAX_DATASET_SIZE = MAX_DATASET_SIZE):
    data = []
    i = 0
    for dirname, _, filenames in os.walk('.\\Data\\anime_faces\\'):
        for filename in filenames:
            #print(os.path.join(dirname, filename))
            data.append(np.asarray(Image.open(os.path.join(dirname, filename)).resize(IMG_SHAPE)))
            if len(data) >= MAX_DATASET_SIZE:
                break
            print(str(i/MAX_DATASET_SIZE) + '%', end = "\r")
            i+=1
    data = np.array(data)
    data = (data  - 127.5) / 127.5
    return data