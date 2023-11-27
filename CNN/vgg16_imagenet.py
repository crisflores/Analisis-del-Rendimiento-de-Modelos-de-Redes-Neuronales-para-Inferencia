
'''   
   ----------------------------------------------------------------------------------------------

   @DRIVER for CNN Profiling: 
   Application to study the performance of convolutional networks. 
   The application implements different convolutional networks layer by layer to generate a 
   performance report for each of them and to observe in detail the bottlenecks of each neural 
   network in the different scenarios. It also generates a report on the system's memory, 
   CPU, GPU, etc. consumption during execution.

   ----------------------------------------------------------------------------------------------

   This program is free software: you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free Software
   Foundation, either version 3 of the License, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
   You should have received a copy of the GNU General Public License along with
   this program. If not, see <http://www.gnu.org/licenses/>.

   ----------------------------------------------------------------------------------------------

   @authors   = "Cristian Flores and Héctor Martínez"
   @contact   = "xcrisflores12@gmail.com and el2mapeh@uco.es
   @copyright = "Copyright 2023, Universidad de Córdoba"
   @license   = "GPLv3"
   @status    = "Production"
   @version   = "1.0"

   ----------------------------------------------------------------------------------------------
'''

import tensorflow as tf
import time
import CNN.cnn_layer as cl

def vgg16_imagenet(input_shape):

    #Timers Initialization
    layers_prof = cl.LayersProfiling()

    #CNN Arquitecture
    cnn = []

    #Create Input Tensor
    inputs = tf.random.normal(input_shape);

    #---------------------------------------------------------------
    # Classes Construction
    #---------------------------------------------------------------
    conv = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
    cnn.append([conv, cl.CONV])

    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])
    
    conv = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
    cnn.append([conv, cl.CONV])

    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])

    maxPool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid')
    cnn.append([maxPool, cl.MAXPOOL])
    
    conv = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')
    cnn.append([conv, cl.CONV])

    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])
    
    conv = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')
    cnn.append([conv, cl.CONV])

    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])
    
    maxPool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))
    cnn.append([maxPool, cl.MAXPOOL])
    
    conv = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')
    cnn.append([conv, cl.CONV])

    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])
    
    conv = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')
    cnn.append([conv, cl.CONV])

    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])

    conv = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')
    cnn.append([conv, cl.CONV])

    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])

    maxPool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))
    cnn.append([maxPool, cl.MAXPOOL])
    
    conv = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')
    cnn.append([conv, cl.CONV])

    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])
    
    conv = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')
    cnn.append([conv, cl.CONV])

    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])

    conv = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')
    cnn.append([conv, cl.CONV])

    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])

    maxPool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))
    cnn.append([maxPool, cl.MAXPOOL])

    
    conv = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')
    cnn.append([conv, cl.CONV])

    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])
    
    conv = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')
    cnn.append([conv, cl.CONV])

    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])

    conv = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')
    cnn.append([conv, cl.CONV])

    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])

    maxPool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))
    cnn.append([maxPool, cl.MAXPOOL])

    #----------------------------------------------------------------
    flatten = tf.keras.layers.Flatten()
    cnn.append([flatten, cl.FLATTEN])

    dense   = tf.keras.layers.Dense(4096)
    cnn.append([dense, cl.FC])
    
    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])

    dropout = tf.keras.layers.Dropout(0.5)
    cnn.append([dropout, cl.DROPOUT])
    
    dense   = tf.keras.layers.Dense(4096)
    cnn.append([dense, cl.FC])

    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])
    
    dropout = tf.keras.layers.Dropout(0.5)
    cnn.append([dropout, cl.DROPOUT])
    
    dense   = tf.keras.layers.Dense(1000)
    cnn.append([dense, cl.FC])

    softmax = tf.keras.layers.Softmax()
    cnn.append([softmax, cl.SOFTMAX])
    #---------------------------------------------------------------

    
    #---------------------------------------------------------------
    # CNN Arquitecture
    #---------------------------------------------------------------
    layers_prof.show_header_profile()
    n_layer = 1
    inputs_orig = None
    for layer, t_layer in cnn:
        t0 = time.time()
        inputs = layer(inputs)
        ttot = time.time() - t0
            
        layers_prof.add_layer(t_layer, ttot)
        layers_prof.show_layer(n_layer, t_layer, inputs.shape)
        n_layer += 1
    #---------------------------------------------------------------

    return layers_prof

