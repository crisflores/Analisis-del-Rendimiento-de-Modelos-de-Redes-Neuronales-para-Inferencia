#include <stdio.h>
#include "./../../../software/blis_new/blis/install/include/blis/blis.h"
#include <signal.h>
//   ./../../../software/blis_new/blis/install/include/blis/libblis.a

//FILE *archivo;

int add(int a, int b){
    return a+b;
}

/*
void signal_handler(int signum) {
    if (signum == SIGSEGV) {
        // Manejar la violación de segmento aquí
        printf("Violación de segmento detectada.\n");
        fclose(archivo);
        exit(EXIT_FAILURE);  // O cualquier acción que desees tomar
    }
}*/


//Implementación
void im2row_plus_gemm(float *rows, int ld, float *in,
	int batch, int height, int width, int channel,
	int oheight, int owidth,
	int kheight, int kwidth, int kchannel, int kfilters,
	int vpadding, int hpadding,
	int vstride, int hstride,
	int vdilation, int hdilation,
	float *out, float *kernel)
{

	int b, x, y, row, kx, ix, ky, iy, c, col;
	int i = 0;
 
	//signal(SIGSEGV, signal_handler);
	
	/*archivo = fopen("archivo.txt", "a");

    if (archivo == NULL) {
        fprintf(stderr, "No se pudo abrir el archivo.\n");
        return;
    }*/
	//printf("tamaño_entrada: %d x %d x %d x %d\n",batch, height, width, channel);
	//printf("tamaño_salida: %d x %d x %d x %d\n", batch, oheight, owidth, kfilters);
	//printf("kernel_size: %d x %d x %d x %d\n", kheight, kwidth, kchannel, kfilters);
	//printf("padding: %d x %d\n", vpadding, hpadding);
	//printf("stride: %d x %d\n", vstride, hstride);
	//printf("dilation: %d x %d\n", vdilation, hdilation);
	//#pragma omp parallel for private(b, x, y, row, kx, ix, ky, iy, c, col)
	for (b = 0; b < batch; b++)
		for (x = 0; x < oheight; x++)
			for (y = 0; y < owidth; y++) {
				row = b * oheight * owidth + x * owidth + y;
				for (kx = 0; kx < kheight; kx++) {
					ix = vstride * x + vdilation * kx - vpadding;
					if (0 <= ix && ix < height)
						for (ky = 0; ky < kwidth; ky++) {
							iy = hstride * y + hdilation * ky - hpadding;
							if (0 <= iy && iy < width)
							for (c = 0; c < channel; c++) {
								col = c * kheight * kwidth + kx * kwidth + ky;
								rows[row * channel * kheight * kwidth + col] = in[((b * height + ix) * width + iy) * channel + c];
							}
						}
				}
			}
	//fprintf(archivo,"terminada convolucion\n\n\n");
	//fclose(archivo);
	int mm = kfilters; //numero de canales de salida // 96
    int nn = oheight * owidth * batch; // 55, 55, 512
    int kk = kheight * kwidth * channel; // 11, 11, 3
    float alphap = 1.0;
    float betap  = 1.0;
    int lda = kfilters; // 96
	int ldb = kheight * kwidth * channel; // 11, 11, 3
 	int ldc = kfilters; // 96	
 	sgemm_("N", "N", &mm, &nn, &kk, &alphap, kernel, &lda, rows, &ldb, &betap, out, &ldc);
}