
#include "../common/book.h"
#include "./utils.h"
// #include "./cuda_kernels.cuh"
#include <cufft.h>
#include <iostream>
#include <complex>


#define NX 32
#define NY 32
#define LX (2 * M_PI)
#define LY (2 * M_PI)

using namespace std;

typedef double     SimPixelType;

int main() {


	SimPixelType *x = new SimPixelType[NX * NY];
	SimPixelType *y = new SimPixelType[NX * NY];
	SimPixelType *vx = new SimPixelType[NX * NY];
	complex<SimPixelType> *out = new complex<SimPixelType>[NX * NY];
	for(int j = 0; j < NY; j++){
	    for(int i = 0; i < NX; i++){
	        x[j * NX + i] = i * LX/NX;
	        y[j * NX + i] = j * LY/NY;
	        vx[j * NX + i] = cos(x[j * NX + i] + y[j * NX + i]);
	    }
	}

	for (int j = 0; j < NY; j++){
	    for (int i = 0; i < NX; i++){
	        // printf("%.3f ", vx[j*NX + i]/(NX*NY));
	        cout << vx[j * NX + i] << " ";
	    }
	    // printf("\n");
	    cout << endl;
	}
	cout << endl;
	SimPixelType *d_vx;
	SimPixelType *d_out;
	cudaMalloc(&d_vx, NX * NY * sizeof(SimPixelType));
	cudaMalloc(&d_out, NX * NY * sizeof(cufftDoubleComplex));
	cudaMemcpy(d_vx, vx, NX * NY * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, out, NX * NY * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

	cufftHandle planr2c;
	cufftHandle planc2r;

	int n[2] = {NX, NY};
	int inembed[] = {NX, NY};
	int onembed[] = {NX, NY};
	int depth = 128;

	/* Forward Fourier Transform plan */
	cufftPlanMany(&planr2c,
	            2, // rank
	            n, // dimension
	            inembed,
	            1, // istride
	            NX * NY, // idist
	            onembed,
	            1, //ostride
	            NX * NY, // odist
	            CUFFT_D2Z,
	            1);


	
	/* Inverse Fourier Transform plan */
	cufftPlanMany(&planc2r,
	            2, // rank
	            n, // dimension
	            onembed,
	            1, // istride
	            NX * NY, // idist
	            inembed,
	            1, //ostride
	            NX * NY, // odist
	            CUFFT_Z2D,
	            1);







	// cufftPlan2d(&planr2c, NY, NX, CUFFT_D2Z);
	// cufftPlan2d(&planc2r, NY, NX, CUFFT_Z2D);
	cufftSetCompatibilityMode(planr2c, CUFFT_COMPATIBILITY_NATIVE);
	cufftSetCompatibilityMode(planc2r, CUFFT_COMPATIBILITY_NATIVE);
	cufftExecD2Z(planr2c, (cufftDoubleReal *)d_vx, (cufftDoubleComplex *)d_out);
	cufftExecZ2D(planc2r, (cufftDoubleComplex *)d_out, (cufftDoubleReal *)d_vx);
	cudaMemcpy(vx, d_vx, NX * NY * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);


	for (int j = 0; j < NY; j++){
	    for (int i = 0; i < NX; i++){
	        // printf("%.3f ", vx[j*NX + i]/(NX*NY));
	        cout << vx[j * NX + i]/( NX * NY) << " ";
	    }
	    // printf("\n");
	    cout << endl;
	}

	return 0;
}