#ifndef SPATIALFILTERING_H_INCLUDE
#define SPATIALFILTERING_H_INCLUDE

#include "tm.cuh"

#define STATIAL_FILTERING_ITERATION 10
#define FILTER_ACCURACY_H 350
#define FILTER_ACCURACY_V 635

float ppp[MAX_HEIGHT][MAX_HEIGHT];
float pixels1[MAX_HEIGHT][MAX_WIDTH];

__global__ void gpuSpatialFilteringH(int row, float *gpuJpixels, int Jpixels_width, float *gpuIpixels, int Ipixels_width, float *gpuHh, int Hh_width, int Height, int Width) {
	const int tid = threadIdx.x;
	float sum;
	extern __shared__ float Jdata[];

	for (int i = tid; i < Width; i += THREAD_NUM) {
		Jdata[i] = gpuJpixels[row * Jpixels_width + i];
	}

	__syncthreads();

	for (int p = tid; p < Width; p += THREAD_NUM) {
		sum = 0;
		for (int q = 0; q <= FILTER_ACCURACY_H + FILTER_ACCURACY_H; q++) {
			if (q + p - FILTER_ACCURACY_H >= 0 && q + p - FILTER_ACCURACY_H < Width)
				sum += gpuHh[q * Hh_width + p] * Jdata[q + p - FILTER_ACCURACY_H];
		}
		gpuJpixels[row * Jpixels_width + p] = 
			sum + gpuHh[FILTER_ACCURACY_H * Hh_width + p] * (gpuIpixels[row * Ipixels_width + p] - Jdata[p]);
	}
}

__global__ void gpuSpatialFilteringV(int col, float *gpuJpixels, int Jpixels_width, float *gpuIpixels, int Ipixels_width, float *gpuHv, int Hv_width, int Height, int Width) {
	const int tid = threadIdx.x;
	float sum;
	extern __shared__ float Jdata[];

	for (int i = tid; i < Height; i += THREAD_NUM) {
		Jdata[i] = gpuJpixels[i * Jpixels_width + col];
	}

	__syncthreads();

	for (int p = tid; p < Height; p += THREAD_NUM) {
		sum = 0; 
		for (int q = 0; q < Height; q++) {
			sum += gpuHv[p * Hv_width + q] * Jdata[q];
		}
		gpuJpixels[p * Jpixels_width + col] =
			sum + gpuHv[p * Hv_width + p] * (gpuIpixels[p * Ipixels_width + col] - Jdata[p]);
	}
}

__global__ void initHH(float *gpuHh, int Hh_width, float *gpuPAIH, int PAIH_width, int Height, int Width) {
	const int tid = threadIdx.x;
	float sum;

	for (int q = tid; q < Width; q += THREAD_NUM) {
		sum = 1;
		for (int n = 1; n <= FILTER_ACCURACY_H; n++) {
			if (q + n < Width) sum += gpuPAIH[(FILTER_ACCURACY_H - n) * PAIH_width + q + n];
			if (q - n >= 0) sum += gpuPAIH[(FILTER_ACCURACY_H + n) * PAIH_width + q - n];
		}
		//assert(sum > 0);
		for (int p = 0; p <= FILTER_ACCURACY_H + FILTER_ACCURACY_H; p++) {
			gpuHh[p * Hh_width + q] = gpuPAIH[p * PAIH_width + q] / sum;
		}
	}
}

__global__ void initHV(float *gpuHv, int Hv_width, float *gpuPAIV, int PAIV_width, int Height, int Width) {
	const int tid = threadIdx.x;
	float sum;
	for (int p = tid; p < Height; p += THREAD_NUM) {
		sum = 1;
		for (int q = p + 1; q < Height; q++) {
			sum += gpuPAIV[p * PAIV_width + q];
		}
		for (int q = p - 1; q >= 0; q--) {
			sum += gpuPAIV[p * PAIV_width + q];
		}
		//assert(sum > 0);
		for (int q = 0; q < Height; q++) {
			gpuHv[p * Hv_width + q] = gpuPAIV[p * PAIV_width + q] / sum;
		}
	}
}

__global__ void initPAIH(int row, float *gpuPAIH, int PAIH_width, float *gpupaiH, int paiH_width, int Height, int Width) {
	const int tid = threadIdx.x;
	extern __shared__ float paidata[];

	for (int i = tid; i < Width; i += THREAD_NUM) {
		paidata[i] = gpupaiH[row * paiH_width + i];
	}

	__syncthreads();

	for (int p = tid; p < Width; p += THREAD_NUM) {
		gpuPAIH[FILTER_ACCURACY_H * PAIH_width + p] = (float)1;
		for (int q = 1; q <= FILTER_ACCURACY_H; q++) {
			if (p + q > Width) 
				gpuPAIH[(FILTER_ACCURACY_H + q) * PAIH_width + p] = (float)0;
			else 
				gpuPAIH[(FILTER_ACCURACY_H + q) * PAIH_width + p] = gpuPAIH[(FILTER_ACCURACY_H + q - 1) * PAIH_width + p] * paidata[p + q - 1];
			if (p - q < 0) 
				gpuPAIH[(FILTER_ACCURACY_H - q) * PAIH_width + p] = (float)0;
			else 
				gpuPAIH[(FILTER_ACCURACY_H - q) * PAIH_width + p] = gpuPAIH[(FILTER_ACCURACY_H - q + 1) * PAIH_width + p] * paidata[p - q];
		}
	}
}

__global__ void initPAIV(int col, float *gpuPAIV, int PAIV_width, float *gpupaiV, int paiV_width, int Height, int Width) {
	const int tid = threadIdx.x;
	extern __shared__ float paidata[];

	for (int i = tid; i < Height; i += THREAD_NUM) {
		paidata[i] = gpupaiV[i * paiV_width + col];
	}

	__syncthreads();

	for (int p = tid; p < Height; p += THREAD_NUM) {
		gpuPAIV[p * PAIV_width + p] = (float)1;
		for (int q = p + 1; q < Height; q++) {
			gpuPAIV[p * PAIV_width + q] = gpuPAIV[p * PAIV_width + q - 1] * paidata[q - 1];
		}
		for (int q = p - 1; q >= 0; q--) {
			gpuPAIV[p * PAIV_width + q] = gpuPAIV[p * PAIV_width + q + 1] * paidata[q];
		}
	}
}

__global__ void initPai(float *gpupaiH, int paiH_width, float *gpupaiV, int paiV_width, float *gpuIpixels, int Ipixels_width, int Height, int Width) {
	const int tid = threadIdx.x;
	float luminance1, luminance2;
	//PaiH
	for (int i = tid; i < Height; i += THREAD_NUM) {
		for (int j = 0; j < Width - 1; j++) {
			luminance1 = gpuIpixels[i * Ipixels_width + j];
			luminance2 = gpuIpixels[i * Ipixels_width + j + 1];
			gpupaiH[i * paiH_width + j] =
				((float)1 / ((float)1 + (luminance1 - luminance2) * (luminance1 - luminance2) / Sigma / Sigma));
		}
		gpupaiH[i * paiH_width + Width - 1] = float(1);
	}
	//PaiV
	for (int j = tid; j < Width; j += THREAD_NUM) {
		for (int i = 0; i < Height - 1; i++) {
			luminance1 = gpuIpixels[i * Ipixels_width + j];
			luminance2 = gpuIpixels[(i + 1) * Ipixels_width + j];
			gpupaiV[i * paiV_width + j] =
				((float)1 / ((float)1 + (luminance1 - luminance2) * (luminance1 - luminance2) / Sigma / Sigma));
		}
		gpupaiV[(Height - 1) * paiV_width + j] = float(1);
	}
}

bool Frame::SpatialFiltering(int FrameId) {
	float *gpuIpixels, *gpuJpixels;
	float *gpupaiH, *gpupaiV;
	float *gpuPAIH, *gpuPAIV;
	float *gpuHh, *gpuHv;

	FILE *fp;
	char name[NAME_SIZE];
	cudaError_t result;

	//initpai
	size_t pitch_Ipixels, pitch_Jpixels, pitch_paiH, pitch_paiV;
	cudaMallocPitch((void**)&gpuIpixels, &pitch_Ipixels, sizeof(float) * MAX_WIDTH, MAX_HEIGHT);
	cudaMallocPitch((void**)&gpuJpixels, &pitch_Jpixels, sizeof(float) * MAX_WIDTH, MAX_HEIGHT);

	//fp = fopen("cache/ipixels.txt", "w");
	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
			pixels[x][y] = ((float)Ipixels[FrameId][x][y].r * 299 + (float)Ipixels[FrameId][x][y].g * 587 + (float)Ipixels[FrameId][x][y].b * 114 + 500) / 1000;
			//fprintf(fp, "%f ", pixels[x][y]);
			pixels[x][y] = log(pixels[x][y] + Offset);
		}
		//fprintf(fp, "\n");
	}
	//fclose(fp);

	cudaMemcpy2D(gpuIpixels, pitch_Ipixels, pixels, sizeof(float) * MAX_WIDTH, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyHostToDevice);
	cudaMemcpy2D(gpuJpixels, pitch_Jpixels, pixels, sizeof(float) * MAX_WIDTH, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyHostToDevice);
	
	cudaMallocPitch((void**)&gpupaiH, &pitch_paiH, sizeof(float) * MAX_WIDTH, MAX_HEIGHT);
	cudaMallocPitch((void**)&gpupaiV, &pitch_paiV, sizeof(float) * MAX_WIDTH, MAX_HEIGHT);

	initPai <<<1, THREAD_NUM, 0 >>>
		(gpupaiH, pitch_paiH / sizeof(float), gpupaiV, pitch_paiV / sizeof(float), gpuIpixels, pitch_Ipixels / sizeof(float), height, width);
	
	//debug
	/*cudaMemcpy2D(pixels, sizeof(float) * MAX_WIDTH, gpupaiH, pitch_paiH, sizeof(float)* MAX_WIDTH, MAX_HEIGHT, cudaMemcpyDeviceToHost);
	BaseLayer[FrameId].resizeErase(height, width);
	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
			BaseLayer[FrameId][x][y].r = exp(pixels[x][y]) - Offset;
		}
	}
	cudaMemcpy2D(pixels, sizeof(float) * MAX_WIDTH, gpupaiV, pitch_paiV, sizeof(float)* MAX_WIDTH, MAX_HEIGHT, cudaMemcpyDeviceToHost);
	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
			BaseLayer[FrameId][x][y].r += exp(pixels[x][y]) - Offset;
			BaseLayer[FrameId][x][y].r /= 2;
			BaseLayer[FrameId][x][y].g = BaseLayer[FrameId][x][y].b = BaseLayer[FrameId][x][y].r;
			BaseLayer[FrameId][x][y].a = Ipixels[FrameId][x][y].a;
		}
	}
	WriteFrame(BaseLayer[FrameId][0], "baselayer/sfpermeability-75.exr");
	return 0;*/

	printf("initpai done.\n");
	
	//debug

	/*BaseLayer[FrameId].resizeErase(height, width);
	result = cudaMemcpy2D(pixels, sizeof(float) * MAX_WIDTH, gpupaiH, pitch_paiH, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyDeviceToHost);
	if (result != cudaSuccess) printf("Error\n");
	fp = fopen("cache/pai_before.txt", "w");
	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
			BaseLayer[FrameId][x][y].r = exp(pixels[x][y]) - Offset;
			BaseLayer[FrameId][x][y].g = BaseLayer[FrameId][x][y].b = BaseLayer[FrameId][x][y].r;
			BaseLayer[FrameId][x][y].a = Ipixels[FrameId][x][y].a;
			fprintf(fp, "%f ", pixels[x][y]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);*/

	//filtering

	size_t pitch_PAIH, pitch_PAIV, pitch_Hh, pitch_Hv;
	cudaMallocPitch((void**)&gpuPAIH, &pitch_PAIH, sizeof(float) * MAX_WIDTH, (FILTER_ACCURACY_H << 1) + 1);
	cudaMallocPitch((void**)&gpuPAIV, &pitch_PAIV, sizeof(float) * MAX_HEIGHT, MAX_HEIGHT);
	cudaMallocPitch((void**)&gpuHh, &pitch_Hh, sizeof(float) * MAX_WIDTH, (FILTER_ACCURACY_H << 1) + 1);
	cudaMallocPitch((void**)&gpuHv, &pitch_Hv, sizeof(float) * MAX_HEIGHT, MAX_HEIGHT);

	for (int it = 1; it <= STATIAL_FILTERING_ITERATION; it++) {
		printf("it = %d\n", it);
		//horizontal
		if (it & 1) {
			for (int x = 0; x < height; x++) {
				//initH
				initPAIH<<<1, THREAD_NUM, pitch_paiH>>>(x, gpuPAIH, pitch_PAIH / sizeof(float), gpupaiH, pitch_paiH / sizeof(float), height, width);
				initHH<<<1, THREAD_NUM, 0>>>(gpuHh, pitch_Hh / sizeof(float), gpuPAIH, pitch_PAIH / sizeof(float), height, width); 
				
				//filtering
				gpuSpatialFilteringH<<<1, THREAD_NUM, pitch_Jpixels>>>
					(x, gpuJpixels, pitch_Jpixels / sizeof(float), gpuIpixels, pitch_Ipixels / sizeof(float), gpuHh, pitch_Hh / sizeof(float), height, width);
				
				/*if (x == 0){
					fp = fopen("cache/719PAIH-new.txt", "w");
					result = cudaMemcpy2D(ppp, sizeof(float)*MAX_WIDTH, gpuPAIH, pitch_PAIH, sizeof(float)*MAX_WIDTH, (FILTER_ACCURACY_H << 1) + 1, cudaMemcpyDeviceToHost);
					if (result != cudaSuccess) printf("Error-1\n");
					else for (int i = 0; i < (FILTER_ACCURACY_H << 1) + 1; i++) {
						for (int j = 0; j < 50; j++) {
							fprintf(fp, "%f ", ppp[i][j]);
						}
						fprintf(fp, "\n");
					}
					fclose(fp);

					sprintf(name, "cache/Jpixels_after%03d.txt", x);
					fp = fopen(name, "w");
					result = cudaMemcpy2D(pixels, sizeof(float)*MAX_WIDTH, gpuJpixels, pitch_Jpixels, sizeof(float)*MAX_WIDTH, MAX_HEIGHT, cudaMemcpyDeviceToHost);
					if (result != cudaSuccess) printf("Error-2\n");
					else for (int i = 0; i < height; i++) {
						for (int j = 0; j < width; j++) {
							fprintf(fp, "%f ", pixels[i][j]);
						}
						fprintf(fp, "\n");
					}
					fclose(fp);
				}*/
			}
		}
		//vertical
		else {

			for (int y = 0; y < width; y++) {
				//initH
				initPAIV <<<1, THREAD_NUM, sizeof(float)*MAX_HEIGHT>>>(y, gpuPAIV, pitch_PAIV / sizeof(float), gpupaiV, pitch_paiV / sizeof(float), height, width);
				initHV<<<1, THREAD_NUM, 0>>>(gpuHv, pitch_Hv / sizeof(float), gpuPAIV, pitch_PAIV / sizeof(float), height, width);

				//filtering
				gpuSpatialFilteringV<<<1, THREAD_NUM, sizeof(float) * MAX_HEIGHT>>>
					(y, gpuJpixels, pitch_Jpixels / sizeof(float), gpuIpixels, pitch_Ipixels / sizeof(float), gpuHv, pitch_Hv / sizeof(float), height, width);

				//result = cudaMemcpy2D(pixels, sizeof(float)*MAX_WIDTH, gpupaiV, pitch_paiV, sizeof(float)*MAX_WIDTH, MAX_HEIGHT, cudaMemcpyDeviceToHost);
				//if (result != cudaSuccess);// printf("error\n");
				//else printf("%f\n", pixels[0][0]);
				/*if (y == 0){
					fp = fopen("cache/000PAIV-new.txt", "w");
					result = cudaMemcpy2D(ppp, sizeof(float)*MAX_HEIGHT, gpuPAIV, pitch_PAIV, sizeof(float)*MAX_HEIGHT, MAX_HEIGHT, cudaMemcpyDeviceToHost);
					if (result != cudaSuccess) printf("Error-1\n");
					else for (int i = 0; i < height; i++) {
						for (int j = 0; j < height; j++) {
							fprintf(fp, "%f ", ppp[i][j]);
						}
						fprintf(fp, "\n");
					}
					fclose(fp);

					sprintf(name, "cache/Jpixels_after%03d.txt", y);
					fp = fopen(name, "w");
					result = cudaMemcpy2D(pixels, sizeof(float)*MAX_WIDTH, gpuJpixels, pitch_Jpixels, sizeof(float)*MAX_WIDTH, MAX_HEIGHT, cudaMemcpyDeviceToHost);
					if (result != cudaSuccess) printf("Error-2\n");
					else for (int i = 0; i < height; i++) {
						for (int j = 0; j < width; j++) {
							fprintf(fp, "%f ", pixels[i][j]);
						}
						fprintf(fp, "\n");
					}
					fclose(fp);
				}*/
			}
		}
		if (it == STATIAL_FILTERING_ITERATION - 1) {
			result = cudaMemcpy2D(pixels1, sizeof(float) * MAX_WIDTH, gpuJpixels, pitch_Jpixels, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyDeviceToHost);
			if (result != cudaSuccess) printf("Error!\n");
		}
	}
	cudaFree(gpupaiH);
	cudaFree(gpuPAIH);
	cudaFree(gpuHh);
	cudaFree(gpupaiV);
	cudaFree(gpuPAIV);
	cudaFree(gpuHv);

	printf("Filtering done.\n");

	//CopyBack
	BaseLayer[FrameId].resizeErase(height, width);
	float luminance;
	//fp = fopen("cache/baselayer-new.txt", "w");

	result = cudaMemcpy2D(pixels, sizeof(float) * MAX_WIDTH, gpuJpixels, pitch_Jpixels, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyDeviceToHost);
	if (result != cudaSuccess) printf("The Lastest Error!\n");

	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
			//luminance = ((float)Ipixels[FrameId][x][y].r * 299 + (float)Ipixels[FrameId][x][y].g * 587 + (float)Ipixels[FrameId][x][y].b * 114 + 500) / 1000;
			//fprintf(fp, "%f\n", pixels[x][y]);
			pixels[x][y] = (pixels[x][y] + pixels1[x][y]) / 2.0;
			pixels[x][y] = exp(pixels[x][y]) - Offset;

			/*BaseLayer[FrameId][x][y].r = pow((Ipixels[FrameId][x][y].r / luminance) , 0.6) * pixels[x][y];
			BaseLayer[FrameId][x][y].g = pow((Ipixels[FrameId][x][y].g / luminance) , 0.6) * pixels[x][y];
			BaseLayer[FrameId][x][y].b = pow((Ipixels[FrameId][x][y].b / luminance) , 0.6) * pixels[x][y];
			BaseLayer[FrameId][x][y].a = Ipixels[FrameId][x][y].a;*/

			BaseLayer[FrameId][x][y].r = BaseLayer[FrameId][x][y].g = BaseLayer[FrameId][x][y].b = pixels[x][y];
			BaseLayer[FrameId][x][y].a = Ipixels[FrameId][x][y].a;

			//fprintf(fp, "%f ", (float)BaseLayer[FrameId][x][y].r);
		}
		//fprintf(fp, "\n");
	}
	//fclose(fp);

	//
	cudaFree(gpuIpixels);
	cudaFree(gpuJpixels);
	return true;
}

#endif //SPATIALFILTERING_H_INCLUDE