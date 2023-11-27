#!/bin/bash

if [[ "$1" == "blis" ]]; then
    THREADS=6

    export OMP_PROC_BIND=TRUE
    export OMP_NUM_THREADS=$THREADS
    # BLIS
    export BLIS_JC_NT=1
    export BLIS_IC_NT=$THREADS
    export BLIS_JR_NT=1
    export BLIS_IR_NT=1

    # Compilar código C
    gcc -shared -o mi_biblioteca.so -fPIC -fopenmp ./miBiblioteca/codigo.c -I./../../software/blis_new/blis/install/include -L./../../software/blis_new/blis/install/lib -lblis
    wait

    # Ejecutar script de Python
    python3 driver_cnn.py AlexNet 1
else
    # Ejecutar script de Python sin BLIS
    python3 driver_cnn.py AlexNet 0
fi
