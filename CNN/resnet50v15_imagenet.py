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

def add_compose_conv2D(cnn, filters, filter_size=(1,1), 
                      strides=(1,1), padding='valid', activation=True):
    
    conv2D = tf.keras.layers.Conv2D(filters, filter_size, strides=strides, padding=padding)
    cnn.append([conv2D, cl.CONV])

    batchNorm = tf.keras.layers.BatchNormalization()
    cnn.append([batchNorm, cl.BATCHNORM])
    
    if activation:
        relu = tf.keras.layers.Activation('relu')
        cnn.append([relu, cl.RELU])

    return cnn

def resnet50v15_imagenet(input_shape):
    #Timers Initialization
    layers_prof = cl.LayersProfiling()

    #CNN Arquitecture
    cnn = []

    #Create Input Tensor (Layer-0)
    inputs = tf.random.normal(input_shape);
    #inputs = tf.keras.layers.Input(shape=input_shape)

    #---------------------------------------------------------------
    # CNN Architecture Construction
    #---------------------------------------------------------------
    n_paths   = (12, 44, 86, 148)

    n_filters = [[64, 64, 256, 256],
                 [64, 64, 256],
                 [64, 64, 256],
                 [128, 128, 512, 512],
                 [128, 128, 512],
                 [128, 128, 512],
                 [128, 128, 512],
                 [256, 256, 1024, 1024],
                 [256, 256, 1024],
                 [256, 256, 1024],
                 [256, 256, 1024],
                 [256, 256, 1024],
                 [256, 256, 1024],
                 [512, 512, 2048, 2048],
                 [512, 512, 2048],
                 [512, 512, 2048]
                 ]

    n_sizes   = [[1, 3, 1, 1],
                 [1, 3, 1],
                 [1, 3, 1],
                 [1, 3, 1, 1],
                 [1, 3, 1],
                 [1, 3, 1],
                 [1, 3, 1],
                 [1, 3, 1, 1],
                 [1, 3, 1],
                 [1, 3, 1],
                 [1, 3, 1],
                 [1, 3, 1],
                 [1, 3, 1],
                 [1, 3, 1, 1],
                 [1, 3, 1],
                 [1, 3, 1]
                 ]
    
    n_strides = [[1, 1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1],
                 [1, 2, 1, 2],
                 [1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1],
                 [1, 2, 1, 2],
                 [1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1],
                 [1, 2, 1, 2],
                 [1, 1, 1],
                 [1, 1, 1]
                 ]

    n_active = [[True, True, False, False], 
                [True, True, False],
                [True, True, False],
                [True, True, False, False],
                [True, True, False],
                [True, True, False],
                [True, True, False],
                [True, True, False, False],
                [True, True, False],
                [True, True, False],
                [True, True, False],
                [True, True, False],
                [True, True, False],
                [True, True, False, False],
                [True, True, False],
                [True, True, False]
                ]

    #---------------------------------------------------------------
    #Block-1
    #---------------------------------------------------------------
    conv2D = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')
    cnn.append([conv2D, cl.CONV])

    maxPool2D = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')
    cnn.append([maxPool2D, cl.MAXPOOL])

    for block in range(0, len(n_filters)):
        cnn.append([None, cl.ADDITION])
        for level in range(0, len(n_filters[block])):
            filters = n_filters[block][level]
            size    = n_sizes[block][level]
            active  = n_active[block][level]
            stride  = n_strides[block][level]
            cnn = add_compose_conv2D(cnn, filters, filter_size=(size, size), 
                                     strides=(stride, stride), padding='same', activation=active)

        relu = tf.keras.layers.Activation('relu')
        cnn.append([relu, cl.RELU])

    avgPool2D = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid')
    cnn.append([avgPool2D, cl.AVERAGEPOOL])

    flatten = tf.keras.layers.Flatten()
    cnn.append([flatten, cl.FLATTEN])
    
    fc = tf.keras.layers.Dense(1000)
    cnn.append([fc, cl.FC])
    
    softmax = tf.keras.layers.Softmax()
    cnn.append([softmax, cl.SOFTMAX])

    #---------------------------------------------------------------
    # CNN Arquitecture
    #---------------------------------------------------------------
    layers_prof.show_header_profile()
    n_layer = 1
    n_path  = 0

    input_path  = None
    inside      = False
    inputs_orig = None
    for layer, t_layer in cnn:
        if t_layer == cl.ADDITION:
            inputs_orig = inputs
        else:
            if n_layer in n_paths: 
                input_path = inputs
                inputs = inputs_orig
                inside = True
                n_path = 0
            t0 = time.time()
            inputs = layer(inputs)
            ttot = time.time() - t0
            layers_prof.add_layer(t_layer, ttot)
            if inside: 
                n_path += 1
                if n_path == 2:
                    inside = False
                    t0 = time.time()
                    inputs = tf.add(input_path, inputs)
                    ttot = time.time() - t0
                    layers_prof.add_layer(cl.ADDITION, ttot)
            

        layers_prof.show_layer(n_layer, t_layer, inputs.shape)
        n_layer += 1
    #---------------------------------------------------------------

    return layers_prof

