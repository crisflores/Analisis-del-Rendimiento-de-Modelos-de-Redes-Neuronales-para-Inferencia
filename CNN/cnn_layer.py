
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

CONV        = 0
MAXPOOL     = 1
BATCHNORM   = 2
ADDITION    = 3
RELU        = 4
AVERAGEPOOL = 5
FLATTEN     = 6
FC          = 7
SOFTMAX     = 8
DENSE       = 9
DROPOUT     = 10

class LayersProfiling:

    
    def __init__(self):
        self.t_layers = {CONV:         0.0, 
                         MAXPOOL:      0.0, 
                         BATCHNORM:    0.0, 
                         ADDITION:     0.0, 
                         RELU:         0.0, 
                         AVERAGEPOOL:  0.0, 
                         FLATTEN:      0.0, 
                         FC:           0.0, 
                         DENSE:        0.0, 
                         DROPOUT:      0.0, 
                         SOFTMAX:      0.0,
                         DENSE:        0.0,
                         DROPOUT:      0.0}

        self.n_layers = {CONV:         0, 
                         MAXPOOL:      0, 
                         BATCHNORM:    0, 
                         ADDITION:     0, 
                         RELU:         0, 
                         AVERAGEPOOL:  0, 
                         FLATTEN:      0, 
                         FC:           0, 
                         DENSE:        0, 
                         DROPOUT:      0, 
                         SOFTMAX:      0,
                         DENSE:        0,
                         DROPOUT:      0}

    def add_layer(self, layer, time):
        self.t_layers[layer] += time
        self.n_layers[layer] += 1

    def layer_to_str(self, layer):
        layer_str = ""

        if   ( CONV        == layer):
            layer_str = "CONVOLUTION"
        elif ( MAXPOOL     == layer):
            layer_str = "MAXPOOL"
        elif ( BATCHNORM   == layer):
            layer_str = "BATCHNORM"
        elif ( ADDITION    == layer):
            layer_str = "ADDITIONBLOCK"
        elif ( RELU        == layer):
            layer_str = "RELU"
        elif ( AVERAGEPOOL == layer):
            layer_str = "AVERAGEPOOL"
        elif ( FLATTEN     == layer):
            layer_str = "FLATTEN"
        elif ( FC          == layer):
            layer_str = "FULLY CONNECT"
        elif ( SOFTMAX     == layer):
            layer_str = "SOFTMAX"
        elif ( DENSE       == layer):
            layer_str = "DENSE"
        elif ( DROPOUT     == layer):
            layer_str = "DROPOUT"
        else: 
            layer_str = "KNOWN"

        return layer_str


    def show_layer(self, n_layer, layer, sizes):
        layer_str = self.layer_to_str(layer)
        sizes_str = ""

        if (len(sizes) != 0):
            sizes_str  = f"{sizes[0]}"
            for size in sizes[1:]:
                sizes_str  = sizes_str + f" x {size}"

        print(f"| {n_layer:3d} |  {layer_str:20s}  |  {sizes_str:20s}  |")
        print(f"+-----+------------------------+------------------------+")

    def show_header_profile(self):
        print(f"+-----+------------------------+------------------------+")
        print(f"| #l  |         LAYER          |  (Batch x H x W x CH)  |")
        print(f"+-----+------------------------+------------------------+")

    def show_profile(self):
        t_total = 0.0
        n_total = 0
        for t in self.t_layers.values():
            t_total += t

        print(f"+--------------------------------------------------------------------------+")
        print(f"|                                CNN LOG PROFILE                           |")
        print(f"+----------------------+-----------------+-----------------+---------------+")
        print(f"| #Layer               |  Total Calls    | Total Time (s)  |  Total Ratio  |")
        print(f"+----------------------+-----------------+-----------------+---------------+")

        for l in self.t_layers:
            _n = self.n_layers[l]
            if _n != 0:
                layer_str = self.layer_to_str(l)
                _t = self.t_layers[l]
                n_total += _n
                print(f"| {layer_str:20s} | {_n:15d} | {_t:15.5f} |  {(_t * 100 / t_total):10.2f}%% |")

                print(f"+----------------------+-----------------+-----------------+---------------+")
        print(f"| TOTAL                | {n_total:15d} | {t_total:15.5f} |         100%% |")
        print(f"+----------------------+-----------------+-----------------+---------------+\n")

