
#Two classes, Autoencoder and Discriminator 

#NOTE THAT THE TRULY ORIGINAL ARCHITECTURE HAD NO DROPOUT IN THE DISCRIMINATOR!!!
import tensorflow as tf
import numpy as np

class Encoder_layer(tf.keras.Model):
    def __init__(self, filters = 16, input_shape = (256, 256, 3)): #input shape: tupel with dimension of input (with attributes)
        super(Encoder_layer, self).__init__()
        self.layer = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
                tf.keras.layers.Conv2D(filters=filters, 
                        kernel_size=(4, 4), 
                        strides=2, 
                        activation='linear', 
                        input_shape=input_shape, 
                        padding='same', 
                        kernel_initializer='he_uniform'),
                tf.keras.layers.BatchNormalization(), 
                tf.keras.layers.LeakyReLU(alpha=0.2),
            ]
        )

    def call(self, x):
        return self.layer(x)

class Decoder_layer(tf.keras.Model):
    def __init__(self, filters = 16, input_shape = (128, 128, 16)): #input shape: tuple with dimension of input (with attributes)
        super(Decoder_layer, self).__init__()
        self.layer = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2DTranspose(filters=filters, 
                        kernel_size=(4, 4), 
                        strides=2, 
                        activation='linear', 
                        input_shape=input_shape, 
                        padding='same', 
                        kernel_initializer='he_uniform'),
                tf.keras.layers.BatchNormalization(), 
                tf.keras.layers.LeakyReLU(alpha=0.2),
            ]
        )
    
    def call(self, x):
        return self.layer(x)


class AutoEncoder(tf.keras.Model):
    def __init__(self, n_attr = 0):
        super(AutoEncoder, self).__init__()
        self.n_attr = n_attr
        self.encoder = tf.keras.Sequential(
            [
            #layer 1
            tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
            tf.keras.layers.Conv2D(filters=16, 
                    kernel_size=(4, 4), 
                    strides=2, 
                    activation='linear', 
                    input_shape=(256, 256, 3), 
                    padding='same', 
                    kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.LeakyReLU(alpha=0.2),
            #layer 2
            tf.keras.layers.Conv2D(filters=32, 
                    kernel_size=(4, 4), 
                    strides=2, 
                    activation='linear', 
                    input_shape=(128, 128, 16), 
                    padding='same', 
                    kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.LeakyReLU(alpha=0.2),
            #layer 3
            tf.keras.layers.Conv2D(filters=64, 
                    kernel_size=(4, 4), 
                    strides=2, 
                    activation='linear', 
                    input_shape=(64, 64, 32), 
                    padding='same', 
                    kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.LeakyReLU(alpha=0.2),
            #layer 4
            tf.keras.layers.Conv2D(filters=128, 
                    kernel_size=(4, 4), 
                    strides=2, 
                    activation='linear', 
                    input_shape=(32, 32, 64), 
                    padding='same', 
                    kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.LeakyReLU(alpha=0.2),
            #layer 5
            tf.keras.layers.Conv2D(filters=256, 
                    kernel_size=(4, 4), 
                    strides=2, 
                    activation='linear', 
                    input_shape=(16, 16, 128), 
                    padding='same', 
                    kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.LeakyReLU(alpha=0.2),
            #layer 6
            tf.keras.layers.Conv2D(filters=512, 
                    kernel_size=(4, 4), 
                    strides=2, 
                    activation='linear', 
                    input_shape=(8, 8, 256), 
                    padding='same', 
                    kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.LeakyReLU(alpha=0.2),
            #layer 7
            tf.keras.layers.Conv2D(filters=512, 
                    kernel_size=(4, 4), 
                    strides=2, 
                    activation='linear', 
                    input_shape=(4, 4, 512), 
                    padding='same', 
                    kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.LeakyReLU(alpha=0.2),
            ]
            )

        self.decoder1 = tf.keras.Sequential([
            #layer 1
            tf.keras.layers.Conv2DTranspose(filters=512, 
                    kernel_size=(4, 4), 
                    strides=2, 
                    activation='linear', 
                    input_shape=(2, 2, 512 + self.n_attr), 
                    padding='same', 
                    kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.LeakyReLU(alpha = 0.2),
            ])
        self.decoder2 = tf.keras.Sequential([
            #layer 2
            tf.keras.layers.Conv2DTranspose(filters=256, 
                    kernel_size=(4, 4), 
                    strides=2, 
                    activation='linear', 
                    input_shape=(4, 4, 512 + self.n_attr), 
                    padding='same', 
                    kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.LeakyReLU(alpha = 0.2),
            ])
        self.decoder3 = tf.keras.Sequential([
            #layer 3
            tf.keras.layers.Conv2DTranspose(filters=128, 
                    kernel_size=(4, 4), 
                    strides=2, 
                    activation='linear', 
                    input_shape=(8, 8, 256 + self.n_attr), 
                    padding='same', 
                    kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.LeakyReLU(alpha = 0.2),
            ])
        self.decoder4 = tf.keras.Sequential([

            #layer 4
            tf.keras.layers.Conv2DTranspose(filters=64, 
                    kernel_size=(4, 4), 
                    strides=2, 
                    activation='linear', 
                    input_shape=(16, 16, 128 + self.n_attr), 
                    padding='same', 
                    kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.LeakyReLU(alpha = 0.2),
            ])
        self.decoder5 = tf.keras.Sequential([
            #layer 5
            tf.keras.layers.Conv2DTranspose(filters=32, 
                    kernel_size=(4, 4), 
                    strides=2, 
                    activation='linear', 
                    input_shape=(32, 32, 64 + self.n_attr), 
                    padding='same', 
                    kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.LeakyReLU(alpha = 0.2),
            ])
        self.decoder6 = tf.keras.Sequential([
            #layer 6
            tf.keras.layers.Conv2DTranspose(filters=16, 
                    kernel_size=(4, 4), 
                    strides=2, 
                    activation='linear', 
                    input_shape=(64, 64, 32 + self.n_attr), 
                    padding='same', 
                    kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.LeakyReLU(alpha = 0.2),
            ])
        self.decoder7 = tf.keras.Sequential([
            #layer 7
            tf.keras.layers.Conv2DTranspose(filters=3, 
                    kernel_size=(4, 4), 
                    strides=2, 
                    activation='linear', 
                    input_shape=(128, 128, 16 + self.n_attr), 
                    padding='same', 
                    kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.LeakyReLU(alpha = 0.2),
            ])


    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def decode(self, z, y = tf.constant(0)):
        # Reshape y to the same shape as z
        y0 = tf.tile(tf.reshape(y, (tf.shape(y)[0], 1, 1, self.n_attr)), multiples=(1,tf.shape(z)[1],tf.shape(z)[2], 1))
        # concate y with z with magic
        z0 = tf.concat([z, y0], -1)
        z1 = self.decoder1(z0)

        y1 = tf.tile(tf.reshape(y, (tf.shape(y)[0], 1, 1, self.n_attr)), multiples=(1,tf.shape(z1)[1],tf.shape(z1)[2], 1))
        z1 = tf.concat([z1, y1], -1)
        z2 = self.decoder2(z1)

        y2 = tf.tile(tf.reshape(y, (tf.shape(y)[0], 1, 1, self.n_attr)), multiples=(1,tf.shape(z2)[1],tf.shape(z2)[2], 1))
        z2 = tf.concat([z2, y2], -1)
        z3 = self.decoder3(z2)

        y3 = tf.tile(tf.reshape(y, (tf.shape(y)[0], 1, 1, self.n_attr)), multiples=(1,tf.shape(z3)[1],tf.shape(z3)[2], 1))
        z3 = tf.concat([z3, y3], -1)
        z4 = self.decoder4(z3)

        y4 = tf.tile(tf.reshape(y, (tf.shape(y)[0], 1, 1, self.n_attr)), multiples=(1,tf.shape(z4)[1],tf.shape(z4)[2], 1))
        z4 = tf.concat([z4, y4], -1)
        z5 = self.decoder5(z4)

        y5 = tf.tile(tf.reshape(y, (tf.shape(y)[0], 1, 1, self.n_attr)), multiples=(1,tf.shape(z5)[1],tf.shape(z5)[2], 1))
        z5 = tf.concat([z5, y5], -1)
        z6 = self.decoder6(z5)
        
        y6 = tf.tile(tf.reshape(y, (tf.shape(y)[0], 1, 1, self.n_attr)), multiples=(1,tf.shape(z6)[1],tf.shape(z6)[2], 1))
        z6 = tf.concat([z6, y6], -1)
        xrc = self.decoder7(z6)

        return xrc

    def call(self, inputs):
        # inputs: List of features and attributes, [x, y]
        x = inputs[0]
        y = inputs[1]
        z = self.encode(x)
        return self.decode(z, y)

class Discriminator(tf.keras.Model):
  def __init__(self, intermediate_dim = 512, n_attr = 80, input_shape = (2, 2, 512)):
    super(Discriminator, self).__init__()
    self.classifier = tf.keras.Sequential(
      [
        #layer 1:
        tf.keras.layers.Conv2D(filters=intermediate_dim, 
                        kernel_size=(4, 4), 
                        strides=2, 
                        activation='linear', 
                        input_shape=input_shape, 
                        padding='same', 
                        kernel_initializer='he_uniform'),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        
        #layer 2
        tf.keras.layers.Dense(
        units=intermediate_dim,
        kernel_initializer='he_uniform'),

        #layer 3
        tf.keras.layers.Dense(
        units=n_attr,
        kernel_initializer='he_uniform'),
      ]
    )

  def call(self, z):
    return self.classifier(z)

#original images (x), latent images (z) and attributes (y)
def AEloss_full(ae_model, x, y, disc_model, lambda_ae, lambda_e):
  z = ae_model.encode(x)
  reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(ae_model.decode(z, y), x)))
  preds = disc_model(z)
  lies = tf.reshape(1 - y, [y.shape[0], int(y.shape[1]/2), 2])
  guess = tf.reshape(preds, [preds.shape[0], int(preds.shape[1]/2), 2])
  cross_entropy = tf.keras.losses.CategoricalCrossentropy() 
  fooling_cross_entropy = cross_entropy(lies, guess)
  return lambda_ae * reconstruction_error + lambda_e * fooling_cross_entropy

def Dloss(disc_model, z, y):
  preds = disc_model(z)
  truth = tf.reshape(y, [y.shape[0], int(y.shape[1]/2), 2])
  guess = tf.reshape(preds, [preds.shape[0], int(preds.shape[1]/2), 2])
  cross_entropy = tf.keras.losses.CategoricalCrossentropy() 
  disc_cross_entropy = cross_entropy(truth, guess)
  return disc_cross_entropy
