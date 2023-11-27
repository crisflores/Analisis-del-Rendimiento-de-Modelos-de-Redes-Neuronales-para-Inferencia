
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
import ctypes
import numpy as np


def alexnet_imagenet(input_shape, blis=0):

    #Timers Initialization
    layers_prof = cl.LayersProfiling()

    #CNN Arquitecture
    cnn = []

    #Create Input Tensor
    inputs = tf.random.normal(input_shape);
    
    miBiblioteca=ctypes.CDLL('./mi_biblioteca.so')

    #---------------------------------------------------------------
    # Classes Construction
    #---------------------------------------------------------------
    conv = tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), padding='valid')
    cnn.append([conv, cl.CONV])

    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])

    maxPool = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))
    cnn.append([maxPool, cl.MAXPOOL])

    conv = tf.keras.layers.Conv2D(256, (5, 5), padding='same')
    cnn.append([conv, cl.CONV])

    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])

    maxPool = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))
    cnn.append([maxPool, cl.MAXPOOL])

    conv = tf.keras.layers.Conv2D(384, (3, 3), padding='same')
    cnn.append([conv, cl.CONV])
   
    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])
   
    conv = tf.keras.layers.Conv2D(384, (3, 3), padding='same')
    cnn.append([conv, cl.CONV])
   
    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])

    conv = tf.keras.layers.Conv2D(256, (3, 3), padding='same')
    cnn.append([conv, cl.CONV])

    relu = tf.keras.layers.Activation('relu')
    cnn.append([relu, cl.RELU])

    maxPool = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))
    cnn.append([maxPool, cl.MAXPOOL])

    #---------------------------------------------------------------
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
    cnn.append([dense, cl.DENSE])
    
    softmax = tf.keras.layers.Softmax()
    cnn.append([softmax, cl.SOFTMAX])
    #---------------------------------------------------------------

    
    #---------------------------------------------------------------
    # CNN Arquitecture
    #---------------------------------------------------------------
    layers_prof.show_header_profile()
    n_layer = 1
    flag_blis = blis
    i = 0
    inputs_orig = None
    for layer, t_layer in cnn:
        layers_prof.show_layer(n_layer, t_layer, inputs.shape)
        if t_layer != 0 or (t_layer == 0 and flag_blis == 0):
            t0 = time.time()
            inputs = layer(inputs)
            ttot = time.time() - t0
        else:
            batch = inputs.shape[0]
            height = inputs.shape[1] 
            width = inputs.shape[2]
            channel = inputs.shape[3]
            
            if i == 0:
                DEXT = np.zeros(shape=(batch*(11*11*3)*(55*55),), dtype=np.float32)
                DEXT_ptr = DEXT.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            if i == 1:
                DEXT = np.zeros(shape=(batch*1500000,), dtype=np.float32)
                DEXT_ptr = DEXT.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            if i == 2:
                DEXT = np.zeros(shape=(batch*331776,), dtype=np.float32)
                DEXT_ptr = DEXT.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            if i == 3:
                DEXT = np.zeros(shape=(batch*497664,), dtype=np.float32)
                DEXT_ptr = DEXT.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            if i == 4:
                DEXT = np.zeros(shape=(batch*497664*2,), dtype=np.float32)
                DEXT_ptr = DEXT.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            D = inputs.numpy().flatten().astype(np.float32)  # Convierte el tensor a un array NumPy y aplana los datos
            in_ptr = D.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            inputs = layer(inputs)
            
            if layer.padding == 'valid':
                vpadding = 0
                hpadding = 0
            else:
                vpadding = int((layer.kernel.shape[0]-1)/2)
                hpadding = int((layer.kernel.shape[1]-1)/2)
                
            oheight = int(((height-layer.kernel_size[0]+2*vpadding)/layer.strides[0])+1)
            owidth = int(((width-layer.kernel_size[1]+2*hpadding)/layer.strides[1])+1)
                
            DEXTK = np.zeros(shape=(inputs.shape[0]*oheight*owidth*layer.filters,), dtype=np.float32)
            DEXTK_ptr = DEXTK.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            K = layer.kernel.numpy().flatten().astype(np.float32)  # Convierte el tensor a un array NumPy y aplana los datos
            kernel_ptr = K.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            t0 = time.time()
            miBiblioteca.im2row_plus_gemm(DEXT_ptr, 0, in_ptr,
                                                              batch, height, width, channel, 
                                                              oheight, owidth, #ho y wo es el tamaño de la imagen de salida // output shape // input shape
                                                              layer.kernel.shape[0], layer.kernel.shape[1], layer.kernel.shape[2], layer.kernel.shape[3],
                                                              0, 0,
                                                              layer.strides[0], layer.strides[1],
                                                              layer.dilation_rate[0], layer.dilation_rate[1],
                                                              DEXTK_ptr, kernel_ptr)
            ttot = time.time() - t0
            
            tensor_size = np.prod(inputs.shape) # Calcula el tamaño total del tensor
            float_array = np.frombuffer(ctypes.string_at(DEXTK_ptr, tensor_size * np.dtype(np.float32).itemsize), dtype=np.float32) # Convierte el puntero de float a un array de NumPy
            tensor = float_array.reshape(inputs.shape) # Reshape el array a la forma del tensor
            tf_tensor = tf.convert_to_tensor(tensor) # Convierte el tensor de NumPy a un tensor de TensorFlow
            inputs = tf_tensor
            flag_blis = 1
            i = i+1

        layers_prof.add_layer(t_layer, ttot)
        #layers_prof.show_layer(n_layer, t_layer, inputs.shape)
        n_layer += 1
    #---------------------------------------------------------------

    return layers_prof

