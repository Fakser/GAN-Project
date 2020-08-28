from src.Gan.Model import *




gan = Gan(data_path = './Data/paintings/', noise_shape=128, train_type='normal' ,discriminator_loss = discriminator_loss, generator_loss = generator_loss, discriminator_type='discriminator')
# gan = GAN(noise_shape=100, discriminator_loss = wasserstein_discriminator_loss, generator_loss = generator_loss)
try:
    os.mkdir('epochs_images')
except Exception as ex:
    print(ex)
gan.train(5000)

