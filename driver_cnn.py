
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

import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import CNN.cnn_layer as cl

import CNN.alexnet_imagenet     as alexnet
import CNN.resnet50v15_imagenet as resnet50
import CNN.vgg16_imagenet       as vgg16


import time
import psutil as ps
import datetime
import numpy as np
import ctypes

from threading import Thread, Event

RESNET50V15 = 0
ALEXNET     = 1
VGG16       = 2

event = Event()


def profiling_function(delay, log_file, memory_info, cpu_info):
    mesure = 0
    memory_used_max       = 0;
    memory_total_used_max = 0;
    cpu_usage_max         = 0;

    #Memory expressed in bytes. Convert to Gbytes at final
    memory_tot_orig  = ps.virtual_memory().total
    memory_used_orig = ps.virtual_memory().used

    while (True):
        #All mesurments here
        tot_used  = ps.virtual_memory().used
        cpu_usage = np.sum(ps.cpu_percent(interval=None, percpu=True))

        if tot_used > memory_total_used_max: memory_total_used_max = tot_used

        only_used = tot_used - memory_used_orig
        if only_used > memory_used_max: memory_used_max = only_used

        now=datetime.datetime.now()
        tot_used  = tot_used  / (1024 * 1024 * 1024)
        only_used = only_used / (1024 * 1024 * 1024)
        
        if cpu_usage > cpu_usage_max:
            cpu_usage_max = cpu_usage

        log_file.write(f"{mesure};{now};{tot_used:.4f};{only_used:.4f};{cpu_usage:.2f}\n")
        mesure += 1

        time.sleep(delay)
        if event.is_set(): break;

    memory_info[0] = memory_tot_orig       / (1024 * 1024 * 1024)
    memory_info[1] = memory_used_max       / (1024 * 1024 * 1024)
    memory_info[2] = memory_total_used_max / (1024 * 1024 * 1024)

    cpu_info[0] = ps.cpu_count() 
    cpu_info[1] = ps.cpu_count(logical=False)
    cpu_info[2] = cpu_usage_max

def get_filename(dir_path):
    count = 0
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    return dir_path + "/run_" + str(count) + ".csv"

#Memory/CPU Usage Profiler 
def main():

    #---- Run configuration ----#
    delay   = 0.5
    batch   = 512
    threads = 6
    #---------------------------#

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # Verifica si TensorFlow está utilizando la GPU
    #np.show_config()
    print("GPUs disponibles:", gpus)
    memoria = ps.virtual_memory()

    memoria_total_gb = memoria.total / (1024 * 1024 * 1024)
    memoria_disponible_gb = memoria.available / (1024 * 1024 * 1024)
    memoria_utilizada_gb = memoria.used / (1024 * 1024 * 1024)

    print(f"Memoria Total: {memoria_total_gb:.2f} GB")
    print(f"Memoria Disponible: {memoria_disponible_gb:.2f} GB")
    print(f"Memoria Utilizada: {memoria_utilizada_gb:.2f} GB")
    print(f"Porcentaje de Memoria Utilizada: {memoria.percent}%")

    if (len(sys.argv) != 2):
        if (len(sys.argv) != 3):
            print(f"USAGE ERROR: python3 {sys.argv[0]:s} <CNN_MODEL>")
            exit(-1)

    layers_log = None
    cnn_id     = -1
    cnn_name   = sys.argv[1]
    blis = sys.argv[2]


    if (cnn_name == "AlexNet" or cnn_name == "Alexnet" or cnn_name == "alexnet"):
        cnn_id = ALEXNET
    elif (cnn_name == "ResNet50v15" or cnn_name == "ResNet50v15" or cnn_name == "Resnet50v15" or cnn_name == "resnet50v15"):
        cnn_id = RESNET50V15
    elif (cnn_name == "VGG16" or cnn_name == "Vgg16" or cnn_name == "vgg16"):
        cnn_id = VGG16
    else:
        print("ERROR: CNN Model unsuported. Models supported ['AlexNet', 'ResNet50v15', 'Vgg16']")
        exit(-1)

    print(f"\nStarting Tensorflow Profiling '{cnn_name:s}'...\n")

    dir_path="Logs"
    filename = get_filename(dir_path)
    log_file = open(filename, "w")
    log_file.write("#Id;Mesurement Time;System Memory Usage (GB);App Memory Usage (GB);CPU Usage (%)\n")

    memory_info = [0, 0, 0]
    cpu_info    = [0, 0, 0]


    #Fix Number of Threads
    tf.config.threading.set_intra_op_parallelism_threads(threads)

    x = Thread(target=profiling_function, args=(delay, log_file, memory_info, cpu_info))
    x.start()

    time.sleep(delay)

    #(Batch_size (N), Hight (H), Width (W), Channels (C)) #NHWC 
    if (cnn_id == ALEXNET):
        input_shape = (batch, 227, 227, 3)
        if blis=="0":
            layers_log = alexnet.alexnet_imagenet(input_shape)
        else:
            layers_log = alexnet.alexnet_imagenet(input_shape,1)
    elif (cnn_id == RESNET50V15):
        input_shape = (batch, 224, 224, 3)
        layers_log = resnet50.resnet50v15_imagenet(input_shape)
    else:
        input_shape = (batch, 224, 224, 3)
        layers_log = vgg16.vgg16_imagenet(input_shape)

    event.set()
        
    x.join()

    print("")
    print("+-------------------------------------------------+")
    print("|                  RUN LOG PROFILE                |")
    print("+-------------------------------------------------+")
    print("|  [*] Output File     | %-24s |" % (filename))
    print("|  [*] Tensor Batch    | %-4d                     |" % (input_shape[0]))
    print("|  [*] Tensor Size     | %-4d x %-4d              |" % (input_shape[1], input_shape[2]))
    print("|  [*] Tensor Channels | %-4d                     |" % (input_shape[3]))
    print("+-------------------------------------------------+\n")

    layers_log.show_profile()

    print("+--------------------------------------------------------------------+ ")
    print("|                          MEMORY LOG PROFILE                        | ")
    print("+--------------------+------------------------+----------------------+ ")
    print("| TOTAL MEMORY (GB)  | TOTAL MEMORY USED (GB) | CNN MEMORY USED (GB) | ")
    print("+--------------------+------------------------+----------------------+ ")
    print("|      %8.3f      |  %8.2f (%6.2f%%)    |  %8.2f(%6.2f%%)   |      " % (memory_info[0], memory_info[2], memory_info[2] * 100 / memory_info[0], memory_info[1], memory_info[1] * 100 / memory_info[0]))
    print("+--------------------+------------------------+----------------------+\n")

    print("+-----------------------------------------------+ ")
    print("|                CPU LOG PROFILE                | ")
    print("+----------------------+------------------------+ ")
    print("| TOTAL CPUS (LOGICAL) |  TOTAL CPUS USAGE (%)  | ")
    print("+----------------------+------------------------+ ")
    print("|       %3d (%3d)      |   %12.2f         |" % (cpu_info[0], cpu_info[1], cpu_info[2]))
    print("+----------------------+------------------------+")

    log_file.close()

if __name__ == '__main__':
    sys.exit(main())  
