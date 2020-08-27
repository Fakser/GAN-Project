from Gan.Model import *

anim_file = 'dcgan_second_train.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('./epochs_images/image*.png')
    filenames = sorted(filenames)
    last = -1
    for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)