from src.controller import  *
from src.Gan.losses import *
from src.data_load import data_load


class Gan(object):
    def __init__(self, data_path = './Data/anime_faces/', img_shape = IMG_SHAPE,  data_dimensions = 3, train_type = 'normal' ,noise_shape = 128, train_data_size = 0.5, batch_size = 256, buffer_size = 60000, generator_loss = generator_loss, discriminator_loss = discriminator_loss, discriminator_type = 'discriminator'):
        
        """GAN Class init function.

          Args:
            data: dataset that GAN is supposed to learn to mimic
            img_shape: first 2 shapes of each image in the dataset, both numbers should be same,  and multiple of 2
            noise_shape: size of the array that will be the GAN input
            batch_size: size of data batch that will be calculated in one step of training 
            buffer_size: A tf.int64 scalar tf.Tensor, representing the number of elements 
                                from this dataset from which the new dataset will sample.
            generator_loss: function that will represent ghow good generator is
            discriminator_loss: function that will represent how good discriminator is
            discriminator_type: default "discriminator" means that GAN will be trained 
                                in default settings with basic loss functions. If you want to use 
                                wasserstein loss, this value should be changed to "critique". 
          """ 
        self.n_epochs = 0
        self.generator = tf.keras.models.Sequential()
        self.discriminator = tf.keras.models.Sequential()
        self.data_dimensions = data_load(path = data_path, MAX_DATASET_SIZE = 1).shape[3]
        
        if img_shape[0]!=img_shape[1] or is_multiple_of_2(img_shape[0]) == False or is_multiple_of_2(img_shape[1]) == False:
            print('dupa')
            return None
        else:
            self.img_shape = img_shape
        
        self.curr_shape = (32,32)
        self.data_path = data_path
        self.noise_shape = noise_shape
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.train_type = train_type
        #self.train_data = copy.deepcopy(data)


        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        
        self.loss = None

        self.num_examples_to_generate = 16

        # We will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_shape])

        self.create_generator()
        if discriminator_type == 'discriminator':
            self.create_discriminator()
        
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
    
    def create_generator(self):
        """
        function that creates model that is supposed to trick 
        our judge that his generated output is real image
        """
        self.generator.add(tf.keras.layers.Dense(512*4*4, use_bias=False, input_shape=(self.noise_shape,)))
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU())
        self.generator.add(tf.keras.layers.Reshape((4, 4, 512)))
        
        self.generator.add(tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert self.generator.output_shape == (None, 8, 8, 256)
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU())

        self.generator.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert self.generator.output_shape == (None, 16, 16, 128)
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU())

        self.generator.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert self.generator.output_shape == (None, 32, 32, 64)
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU())
        if self.train_type == 'normal':
            for i in range(int(math.log(self.img_shape[0], 2) - 5) - 1):
                self.generator.add(tf.keras.layers.Conv2DTranspose(int(64/((i+1)*2)), (5, 5), strides=(2, 2), padding='same', use_bias=False))
                assert self.generator.output_shape == (None, 32 * (i+1) * 2, 32* (i+1) * 2, (64/((i+1)*2)))
                self.generator.add(tf.keras.layers.BatchNormalization())
                self.generator.add(tf.keras.layers.LeakyReLU())
            
            self.generator.add(tf.keras.layers.Conv2DTranspose(self.data_dimensions, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
            assert self.generator.output_shape == (None, self.img_shape[0], self.img_shape[0], self.data_dimensions)
            self.curr_shape = self.img_shape
        else:
            self.generator.add(tf.keras.layers.Conv2DTranspose(self.data_dimensions, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
            assert self.generator.output_shape == (None, self.curr_shape[0], self.curr_shape[0], self.data_dimensions)
            
        
    def create_discriminator(self, out_activation = 'sigmoid'):
        """
        depending on loss function, judge works like discriminator, or like critique
        """
        self.discriminator.add(tf.keras.layers.Conv2D(self.data_dimensions, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[self.curr_shape[0],self.curr_shape[1], self.data_dimensions]))
        self.discriminator.add(tf.keras.layers.LeakyReLU())
        self.discriminator.add(tf.keras.layers.Dropout(0.3))

        self.discriminator.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.discriminator.add(tf.keras.layers.LeakyReLU())
        self.discriminator.add(tf.keras.layers.Dropout(0.3))
        
        for i in range(int(math.log(self.img_shape[0], 2) - 5) - 1):
            self.discriminator.add(tf.keras.layers.Conv2D(128*(i+1), (5, 5), strides=(2, 2), padding='same'))
            self.discriminator.add(tf.keras.layers.LeakyReLU())
            self.discriminator.add(tf.keras.layers.Dropout(0.3))

        self.discriminator.add(tf.keras.layers.Flatten())
        self.discriminator.add(tf.keras.layers.Dense(1)) #, activation = out_activation))
        tf.keras.utils.plot_model(self.discriminator, to_file = 'discriminator.png')
    
    def increase_resolution(self):
        self.curr_shape = (self.curr_shape[0]*2, self.curr_shape[1]*2)
        
        self.generator.layers[0].activation = tf.keras.activations.linear
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU())

        self.generator.add(tf.keras.layers.Conv2DTranspose(int(32/(2*int(math.log(self.curr_shape[0], 2) - 5))), (5, 5), strides=(2, 2), padding='same', use_bias=False))
        print(self.generator.output_shape)
        assert self.generator.output_shape == (None, self.curr_shape[0], self.curr_shape[1], int(32/(2*int(math.log(self.curr_shape[0], 2) - 5))))
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU())

        self.generator.add(tf.keras.layers.Conv2DTranspose(self.data_dimensions, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
        print(self.generator.output_shape)
        assert self.generator.output_shape == (None, self.curr_shape[0], self.curr_shape[1], self.data_dimensions)

        new_d = tf.keras.models.Sequential()
        new_d.add(tf.keras.layers.Conv2D(self.data_dimensions, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[self.curr_shape[0],self.curr_shape[1], self.data_dimensions]))
        new_d.add(tf.keras.layers.LeakyReLU())
        new_d.add(tf.keras.layers.Dropout(0.3))
        for layer in self.discriminator.layers:
            new_d.add(layer)
        self.discriminator = new_d
    
    def generate_and_save_images(self, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = self.generator(test_input, training=False)
        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            if self.data_dimensions == 1:
                plt.imshow(predictions[i,:,:,0]* 127.5 + 127.5, cmap = 'gray')
            else:
                plt.imshow(predictions[i,:,:,:]/2 + 0.5)
            plt.axis('off')

        plt.savefig('.\\epochs_images\\image_at_epoch_{:04d}.png'.format(epoch))
        #print(self.loss)
        #plt.show()


    def train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_shape])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
            self.loss = [gen_loss, disc_loss]

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, epochs, last_epoch = 0):
        if self.train_type == 'normal':
            train_data = data_load(path = self.data_path, IMG_SHAPE = self.curr_shape)
            batched_data = tf.data.Dataset.from_tensor_slices(train_data).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder = True)
            del train_data
            for epoch in range(epochs):
                start = time.time()
                for image_batch in batched_data:
                    self.train_step(image_batch)

                # Produce images for the GIF as we go
                display.clear_output(wait=True)
                self.generate_and_save_images(epoch + last_epoch,
                                            self.seed)

                # Save the model every 15 epochs
                if (epoch + 1) % 15 == 0:
                    self.checkpoint.save(file_prefix = self.checkpoint_prefix)
                self.n_epochs += 1
                print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start), 'current images shape: ', self.curr_shape)
            self.checkpoint.save(file_prefix = self.checkpoint_prefix)
        else:
            n_of_increases = int(math.log(self.img_shape[0], 2) - 5) + 1
            for increase in range(n_of_increases):
                train_data = data_load(path = self.data_path, IMG_SHAPE = self.curr_shape)
                batched_data = tf.data.Dataset.from_tensor_slices(train_data).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder = True)
                del train_data
                for epoch in range(int(epochs/n_of_increases)):
                    start = time.time()
                    for image_batch in batched_data:
                        self.train_step(image_batch)

                    # Produce images for the GIF as we go
                    display.clear_output(wait=True)
                    self.generate_and_save_images(epoch + increase * int(epochs/n_of_increases) + 1,
                                                self.seed)

                    # Save the model every 15 epochs
                    if (epoch + 1) % 100 == 0:
                        self.checkpoint.save(file_prefix = self.checkpoint_prefix)
                    self.n_epochs += 1
                    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start), 'current images shape: ', self.curr_shape)
                self.increase_resolution()
        
    def restore_checkpoint(self):
        """
        function that restores latest checkpoint
        """
        
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
    
    def generate_random_image(self):
        noise = tf.random.normal([1, self.noise_shape])
        generated_image = self.generator(noise, training=False)
        if self.data_dimensions == 1:
            plt.imshow(generated_image[0,:,:,0]* 127.5 + 127.5, cmap = 'gray')
        else:
            plt.imshow(generated_image[0,:,:,:]/2 + 0.5)
        plt.savefig('.\\ranodm_image.png')
            

