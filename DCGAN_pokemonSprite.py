import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

"""Prevent error while using GPU"""
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print('TensorFlow version:',tf.__version__)

IMG_PATH ='./pokemonGen7Sprite/'

# Reshape to (64,64,3)
IMG_DIM = (64,64,3)
BATCH_SIZE = 128
BATCH_PER_EPOCH = np.floor(1076 / BATCH_SIZE).astype(int)
EPOCHS = 10000
LEARNING_RATE_DISC = 0.00001
LEARNING_RATE_GEN  = 0.0005
random_normal_dimensions = 128
KERNEL_SIZE = (4,4)

# Network layers of discriminator and generator
PARAMETERS_DISC = [128,256]
PARAMETERS_GEN = [128,64]

gen_activation = 'tanh'

AUTOTUNE = tf.data.AUTOTUNE

# Load dataset from directory
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  IMG_PATH,
  label_mode=None,
  image_size=(IMG_DIM[0], IMG_DIM[1])
    ,batch_size=BATCH_SIZE
  )

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1), # Rescale input to range [-1,1]
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomContrast(0.2),
])

train_ds = train_ds.map(data_augmentation, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

def discriminator():
    model = tf.keras.Sequential()
    activation = layers.LeakyReLU(0.2)
    
    firstLayer = True
    for PARAMETER in PARAMETERS_DISC:
        if firstLayer:
            model.add(layers.Conv2D(PARAMETER,KERNEL_SIZE, strides=(2, 2),padding='same', input_shape=(64, 64, 3)))
            model.add(activation)
            firstLayer=False
        else:
            model.add(layers.Conv2D(PARAMETER,KERNEL_SIZE, strides=(2, 2),padding='same'))
            model.add(activation)
    
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))
    
    return model

discriminator = discriminator()
discriminator.summary()

def generator():
    model = tf.keras.Sequential()
    activation = layers.ReLU()

    BOTTLE_DIM = IMG_DIM[0]//(2**len(PARAMETERS_GEN))
    BOTTLENECK = (BOTTLE_DIM)**2*256
    
    model.add(layers.Dense(BOTTLENECK,input_shape=(random_normal_dimensions,)))
    model.add(activation)
    
    model.add(layers.Reshape((BOTTLE_DIM, BOTTLE_DIM, 256)))
    
    for PARAMETER in PARAMETERS_GEN:
        model.add(layers.Conv2DTranspose(PARAMETER,KERNEL_SIZE, strides=(2, 2),padding='same'))
        model.add(activation)
    
    model.add(layers.Conv2D(IMG_DIM[-1],KERNEL_SIZE,padding='same'))
    model.add(layers.Activation(gen_activation))
    
    return model

generator = generator()
generator.summary()

gan = tf.keras.models.Sequential([generator, discriminator])
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE_DISC))
discriminator.trainable = False
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE_GEN))

def plot_results(images, n_cols=None):
    '''visualizes fake images'''
    
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1

    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)

    plt.figure(figsize=(n_cols, n_rows))
    
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(((image+1) /2))
        plt.axis("off")

generator, discriminator = gan.layers
for epoch in range(EPOCHS):
    train_dataset = iter(train_ds)
    print("Epoch {}/{}".format(epoch + 1, EPOCHS))
    dis_loss_epoch = []
    gen_loss_epoch = []
    
    for _ in range(BATCH_PER_EPOCH):
        real_images = next(train_dataset)
        
        noise = tf.random.normal(shape=[BATCH_SIZE, random_normal_dimensions])

        fake_images = generator(noise)

        mixed_images = tf.concat([fake_images, real_images], axis=0)

        # Create labels for discriminator
        discriminator_labels = tf.constant([[0.]] * BATCH_SIZE + [[1.]] * BATCH_SIZE)
        
        discriminator.trainable = True

        dis_loss = discriminator.train_on_batch(mixed_images, discriminator_labels)
        dis_loss_epoch.append(dis_loss)

        noise = tf.random.normal(shape=[BATCH_SIZE, random_normal_dimensions])
        
        generator_labels = tf.constant([[1.]] * BATCH_SIZE)
        
        discriminator.trainable = False

        gen_loss = gan.train_on_batch(noise, generator_labels)
        gen_loss_epoch.append(gen_loss)
    
    if epoch%5==0:
        # Plot results
        plot_results(fake_images[:16], 4)                     
        plt.show()  
    
    print('Discriminator loss=',round(np.mean(dis_loss_epoch),5))    
    print('Generator loss=',round(np.mean(gen_loss_epoch),5))
