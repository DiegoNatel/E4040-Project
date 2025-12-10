# ------------------------------------------------------------------------------
# File: model.py
# Author: Bolun Li (UNI: bl3147)
# Note: All code in this file is my own implementation for the course project,
#       except for externally cited sources.
# ------------------------------------------------------------------------------

"""
model.py

- Construct generator
- Construct discriminator

The main functions of DCGAN 

The original TensorFlow implementation of the GAN model can be found at the link below:
https://www.tensorflow.org/tutorials/generative/dcgan?utm_source=chatgpt.com
The DCGAN model in this project is modified according to the improvements described in the DCGAN paper.

"""

import tensorflow as tf
from tensorflow.keras import layers

WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


# As mentioned in the paper: "Directly applying batchnorm to all layers however resulted in sample oscillation and model instability;
# So we avoid this by not applying batchnorm to the generator output layer and the discriminator input layer.


### Generator of DCGAN
## 0. the architecture refers to Figure 1 in the paper
## 1. replace pooling layers with fractional-strided convolutions(deconvoluntion layer)
## 2. remove fully connected hidden layers for deeper architecture
## 3. use batchnorm
## 4. use ReLU for all layers except output layer; uses Tanh in output layer

class Generator(tf.keras.Model):
    def __init__(self, z_dim = 100):   # 100 dimensional Z (input noise)
        super(Generator, self).__init__()
        self.z_dim = z_dim

        ## full-connected layer: z -> 4×4×1024
        self.dense = layers.Dense(
            4 * 4 * 1024,
            use_bias=False,
            input_shape=(self.z_dim,),
            kernel_initializer=WEIGHT_INIT
        )  # input layer (project)
        self.bn0 = layers.BatchNormalization()   # implement batchnorm
        self.relu0 = layers.ReLU()

        ## transposed convolution layer (deconvolution)
        ## the task of generator is to generate an image from a small feature map, so we need upsampling using deconvolution
        ## four deconvolution layers with kernel_size=5*5

        # deconvolution layer 1: 4×4×1024 → 8×8×512
        self.deconv1 = layers.Conv2DTranspose(
            512, (5,5), strides=(2,2), padding='same', use_bias=False, kernel_initializer=WEIGHT_INIT
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        # deconvolution layer 2: 8×8×512 → 16×16×256
        self.deconv2 = layers.Conv2DTranspose(
            256, (5,5), strides=(2,2), padding='same', use_bias=False, kernel_initializer=WEIGHT_INIT
        )
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

        # deconvolution layer 3: 16×16×256 → 32×32×128
        self.deconv3 = layers.Conv2DTranspose(
            128, (5,5), strides=(2,2), padding='same', use_bias=False, kernel_initializer=WEIGHT_INIT
        )
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.ReLU()

        # deconvolution layer 3 (output layer, use 'tanh' as activation function, no bn layer): 32×32×128 → 64×64×3
        self.deconv4 = layers.Conv2DTranspose(
            3, (5,5), strides=(2,2), padding='same', use_bias=False,
            activation='tanh', kernel_initializer=WEIGHT_INIT
        )
    

    # forward process
    def call(self, z, training=True):

        # input layer:
        x = self.dense(z)
        x = self.bn0(x, training=training)
        x = self.relu0(x)

        # reshape to 4*4*1024
        x = tf.reshape(x, (-1, 4, 4, 1024))

        # hidden layers: (3 deconvolution layers)
        x = self.deconv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.deconv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)

        x = self.deconv3(x)
        x = self.bn3(x, training=training)
        x = self.relu3(x)

        # output layer: [-1,1]
        x = self.deconv4(x)
        return x
    

### Discriminator of DCGAN
## 1. replace pooling layers with strided convolutions
## 2. use batchnorm
## 3. remove fully connected hidden layers for deeper architecture
## 4. use LeakyReLU for all layers

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        ## 4 convolution layers: convolution -> (bn) -> LeakyReLU(set slope=0.2)

        # convolution layer 1 (no bn layer)
        self.conv1 = layers.Conv2D(
            64, (5,5), strides=(2,2), padding='same', use_bias=False,
            kernel_initializer=WEIGHT_INIT
        )
        self.leakyrelu1 = layers.LeakyReLU(alpha=0.2) 

        # convolution layer 2: 64×64×64 → 32×32×128
        self.conv2 = layers.Conv2D(
            128, (5,5), strides=(2,2), padding='same', use_bias=False,
            kernel_initializer=WEIGHT_INIT
        )
        self.bn2 = layers.BatchNormalization()
        self.leakyrelu2 = layers.LeakyReLU(alpha=0.2)

        # convolution layer 3: 32×32×128 → 16×16×256
        self.conv3 = layers.Conv2D(
            256, (5,5), strides=(2,2), padding='same', use_bias=False,
            kernel_initializer=WEIGHT_INIT
        )
        self.bn3 = layers.BatchNormalization()
        self.leakyrelu3 = layers.LeakyReLU(alpha=0.2)

        # convolution layer 4: 16×16×256 → 8×8×512
        self.conv4 = layers.Conv2D(
            512, (5,5), strides=(2,2), padding='same', use_bias=False,
            kernel_initializer=WEIGHT_INIT
        )
        self.bn4 = layers.BatchNormalization()
        self.leakyrelu4 = layers.LeakyReLU(alpha=0.2)

        ## output layer: 8×8×512 → 1 (True or False)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, kernel_initializer=WEIGHT_INIT) 


    def call(self, x, training=True):
        # convolution 1
        x = self.conv1(x)
        x = self.leakyrelu1(x)   # no batchnorm in this layer

        # convolution 2
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.leakyrelu2(x)

        # convolution 3
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.leakyrelu3(x)

        # convolution 4
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.leakyrelu4(x)

        # output layer
        x = self.flatten(x)
        x = self.dense(x)

        return x
    




'''
Date: <09/12/2025>
Written by: <Ziyi Zhao> <zz3459@columbia.edu>

The code in this file was fully implemented by the student.
It has not been generated by AI tools, and it has not been copied 
from any external resource.
'''

# ADDED FOR TASK 3 (CIFAR-10 / ImageNet 32x32)
# ==============================================================================
# The following classes are adapted specifically for 32x32 resolution datasets.
# They follow the same DCGAN architectural guidelines as in the paper but with adjustment(reduce model depth) to match the target input&output dimensions.

class Generator32(tf.keras.Model):
    """
    Generator for 32x32 images.
    Architecture: Z(100) -> 4x4x512 -> 8x8x256 -> 16x16x128 -> 32x32x3
    """
    def __init__(self, z_dim=100):
        super(Generator32, self).__init__()
        self.z_dim = z_dim

        # 1. Project and Reshape
        # We start with 4x4 spatial dimensions. 
        # Using 512 filters allows for sufficient capacity before upsampling.
        # No fully connected hidden layers [cite: 88]
        self.dense = layers.Dense(
            4 * 4 * 512, use_bias=False, input_shape=(self.z_dim,),
            kernel_initializer=WEIGHT_INIT
        )
        self.bn0 = layers.BatchNormalization()
        self.relu0 = layers.ReLU() # ReLU in generator [cite: 89]

        # 2. Deconv Layer 1: 4x4 -> 8x8
        self.deconv1 = layers.Conv2DTranspose(
            256, (5, 5), strides=(2, 2), padding='same', use_bias=False,
            kernel_initializer=WEIGHT_INIT
        )
        self.bn1 = layers.BatchNormalization() # Batchnorm in generator [cite: 87]
        self.relu1 = layers.ReLU()

        # 3. Deconv Layer 2: 8x8 -> 16x16
        self.deconv2 = layers.Conv2DTranspose(
            128, (5, 5), strides=(2, 2), padding='same', use_bias=False,
            kernel_initializer=WEIGHT_INIT
        )
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

        # 4. Output Layer: 16x16 -> 32x32
        # Tanh activation for output layer and No BatchNormalization on output layer
        self.deconv3 = layers.Conv2DTranspose(
            3, (5, 5), strides=(2, 2), padding='same', use_bias=False,
            activation='tanh', kernel_initializer=WEIGHT_INIT
        )

    def call(self, z, training=True):
        x = self.dense(z)
        x = self.bn0(x, training=training)
        x = self.relu0(x)
        
        # Reshape to (Batch, 4, 4, 512)
        x = tf.reshape(x, (-1, 4, 4, 512))

        x = self.deconv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.deconv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)

        # Output is 32x32x3
        x = self.deconv3(x) 
        return x


'''
Date: <09/12/2025>
Written by: <Ziyi Zhao> <zz3459@columbia.edu>

The code in this file was fully implemented by the student.
It has not been generated by AI tools, and it has not been copied 
from any external resource.
'''

class Discriminator32(tf.keras.Model):
    """
    Discriminator for 32x32 images.
    Architecture: 32x32x3 -> 16x16x64 -> 8x8x128 -> 4x4x256 -> Output
    """
    def __init__(self):
        super(Discriminator32, self).__init__()

        # Conv Layer 1: 32x32 -> 16x16
        # No BatchNormalization on discriminator input layer and Use LeakyReLU for all layers in discriminator
        self.conv1 = layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding='same', input_shape=[32, 32, 3],
            kernel_initializer=WEIGHT_INIT
        )
        self.leakyrelu1 = layers.LeakyReLU(alpha=0.2) # Slope 0.2 [cite: 96]

        # Conv Layer 2: 16x16 -> 8x8
        # Use strided convolutions instead of pooling [cite: 86]
        self.conv2 = layers.Conv2D(
            128, (5, 5), strides=(2, 2), padding='same', use_bias=False,
            kernel_initializer=WEIGHT_INIT
        )
        self.bn2 = layers.BatchNormalization()
        self.leakyrelu2 = layers.LeakyReLU(alpha=0.2)

        # Conv Layer 3: 8x8 -> 4x4
        self.conv3 = layers.Conv2D(
            256, (5, 5), strides=(2, 2), padding='same', use_bias=False,
            kernel_initializer=WEIGHT_INIT
        )
        self.bn3 = layers.BatchNormalization()
        self.leakyrelu3 = layers.LeakyReLU(alpha=0.2)

        # Output Layer
        # Flatten and project to scalar (Real/Fake)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, kernel_initializer=WEIGHT_INIT)

    def call(self, x, training=True):
        x = self.conv1(x)
        x = self.leakyrelu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.leakyrelu2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.leakyrelu3(x)

        x = self.flatten(x)
        x = self.dense(x)
        return x