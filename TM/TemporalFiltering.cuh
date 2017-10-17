#ifndef TEMPORALFILTERING_H_INCLUDE
#define TEMPORALFILTERING_H_INCLUDE

#include "tm.cuh"

#define ALLNEIGHB ((NEIGHBORHOOD << 1) + 1)

float p[MAX_FRAME], pp[MAX_FRAME][MAX_FRAME];


//__global__ void gpuTemporalFiltering(float *gpuJpixels, float *gpuIpixels, float *gpuH, int H_width, int iFrame) {
//	const int tid = threadIdx.x;
//	float sum = 0.0;
//	
//	extern __shared__ float Jdata[];
//	Jdata[tid] = gpuJpixels[tid];
//	__syncthreads();
//
//	for (int q = 0; q < iFrame; q++) {
//		sum += gpuH[tid * H_width + q] * Jdata[q];
//	}
//	gpuJpixels[tid] = sum + gpuH[tid * H_width + tid] * (gpuIpixels[tid] - Jdata[tid]);
//}

// (now, forwardlimit - backwardlimit + 1, gpuIpixels, gpuJpixels, pitch_Jpixels / sizeof(float), gpuH, pitch_H, width);
__global__ void gpuTemporalFiltering(int x, int now, int limit, float *gpuIpixels, float *gpuJpixels, float *gpuH, int width) {
	const int tid = threadIdx.x;
	float sum = 0.0;

	/*extern __shared__ float Idata[];
	for (int t = tid; t < width; t += THREAD_NUM) {
	Idata[t] = gpuIpixels[t];
	}
	__syncthreads();*/

	for (int t = tid; t < width; t += THREAD_NUM) {
		sum = 0.0;
		for (int q = 0; q < limit; q++) {
			sum += (gpuH[q * MAX_WIDTH + t] * gpuJpixels[q * MAX_WIDTH + t]);
		}
		gpuIpixels[t] = sum + gpuH[now * MAX_WIDTH + t] * (gpuIpixels[t] - gpuJpixels[now * MAX_WIDTH + t]);

		//assert(gpuIpixels[t] >= 0);
	}
}

__global__ void initH(int now, int limit, float *gpuH, float *gpuPAI, int width) {
	const int tid = threadIdx.x;
	float sum = 0.0;

	for (int t = tid; t < width; t += THREAD_NUM) {
		sum = 0.0;
		for (int q = 0; q < limit; q++) {
			sum += gpuPAI[now * ALLNEIGHB * MAX_WIDTH + q * MAX_WIDTH + t];
		}
		for (int q = 0; q < limit; q++) {
			gpuH[q * MAX_WIDTH + t] = gpuPAI[now * ALLNEIGHB * MAX_WIDTH + q * MAX_WIDTH + t] / sum;
			//assert(gpu[q * MAX_WIDTH + t] >= 0);
		}
	}

	/*for (int t = tid; t < width; t += THREAD_NUM) {
	for (int q = 0; q < limit; q++) {
	sum = 0.0;
	for (int n = 0; n < limit; n++) {
	sum += gpuPAI[n * ALLNEIGHB * MAX_WIDTH + q * MAX_WIDTH + t];
	}
	gpuH[q * MAX_WIDTH + t] = gpuPAI[now * ALLNEIGHB * MAX_WIDTH + q * MAX_WIDTH + t] / sum;
	}
	}*/
}

__global__ void initPAI(int limit, float *gpuPAI, float *gpupai, int width) {
	const int tid = threadIdx.x;

	//!!!
	for (int t = tid; t < width; t += THREAD_NUM) {
		for (int p = 0; p < limit; p++) {
			gpuPAI[p * ALLNEIGHB * MAX_WIDTH + p * MAX_WIDTH + t] = 1.f;
			for (int q = p + 1; q < limit; q++) {
				gpuPAI[p * ALLNEIGHB * MAX_WIDTH + q * MAX_WIDTH + t] = gpuPAI[p * ALLNEIGHB * MAX_WIDTH + (q - 1) * MAX_WIDTH + t] * gpupai[(q - 1) * MAX_WIDTH + t];
			}
			for (int q = p - 1; q >= 0; q--) {
				gpuPAI[p * ALLNEIGHB * MAX_WIDTH + q * MAX_WIDTH + t] = gpuPAI[p * ALLNEIGHB * MAX_WIDTH + (q + 1) * MAX_WIDTH + t] * gpupai[q * MAX_WIDTH + t];
				//assert(p * ALLNEIGHB * MAX_WIDTH + q * MAX_WIDTH + t < ALLNEIGHB * ALLNEIGHB * MAX_WIDTH);
			}
		}
	}
}

__global__ void initpai(int limit, float *gpuJpixels, float *gpupai, int width) {
	const int tid = threadIdx.x;

	float pai1, pai2, luminance1, luminance2, luminance;
	for (int t = tid; t < width; t += THREAD_NUM) {
		for (int i = 0; i < limit - 1; i++) {
			luminance1 = gpuJpixels[i * MAX_WIDTH + t];
			luminance2 = gpuJpixels[(i + 1) * MAX_WIDTH + t];
			luminance = luminance1 - luminance2;

			pai1 = (float)1 / ((float)1 + luminance * luminance / PHOTO_CONSTANCY_Sigma / PHOTO_CONSTANCY_Sigma);
			pai2 = (float)1 / ((float)1 + luminance * luminance / FLOW_GRADIENT_Sigma / FLOW_GRADIENT_Sigma);
			gpupai[i * MAX_WIDTH + t] = pai1 * pai2;
			//!!!
			//assert(gpupai[i * MAX_WIDTH + t] >= 0 || gpupai[i * MAX_WIDTH + t] < 0);
		}
		gpupai[(limit - 1) * MAX_WIDTH + t] = (float)1;
	}
}

float jpixels[ALLNEIGHB * ALLNEIGHB][MAX_WIDTH];

inline
bool Frame::TemporalFiltering(int iframe) {

	int backwardlimit = 0 > iframe - NEIGHBORHOOD ? 0 : iframe - NEIGHBORHOOD;
	int forwardlimit = iframe + NEIGHBORHOOD < N - 1 ? iframe + NEIGHBORHOOD : N - 1;
	int now;
	char name[NAME_SIZE];

	BaseLayer[iframe].resizeErase(height, width);
	DetailLayer[iframe].resizeErase(height, width);
	//!!!
	sprintf(name, "hallway/clip_000008.%06d.exr", iframe + 1);
	//sprintf(name, "result/spatialfiler%03d-10-300.exr", iframe+1);
	ReadFrame(iframe, name, 2);

	float *gpupai, *gpuPAI, *gpuH;
	float *gpuIpixels, *gpuJpixels;
	float ipixels[MAX_WIDTH];
	cudaError result;
	FILE *fp;

	size_t pitch_pai, pitch_PAI, pitch_H, pitch_Jpixels;
	cudaMalloc((void**)&gpuIpixels, sizeof(float) * MAX_WIDTH);

	//cudaMallocPitch((void**)&gpupai, &pitch_pai, sizeof(float) * MAX_WIDTH, ALLNEIGHB);
	//cudaMallocPitch((void**)&gpuJpixels, &pitch_Jpixels, sizeof(float) * MAX_WIDTH, ALLNEIGHB);
	cudaMalloc((void**)&gpuJpixels, sizeof(float) * ALLNEIGHB * MAX_WIDTH);
	cudaMalloc((void**)&gpupai, sizeof(float) * ALLNEIGHB * MAX_WIDTH);
	cudaMalloc((void**)&gpuH, sizeof(float) * ALLNEIGHB * MAX_WIDTH);
	//cudaMallocPitch((void**)&gpuH, &pitch_H, sizeof(float) * MAX_WIDTH, ALLNEIGHB);

	cudaMalloc((void**)&gpuPAI, sizeof(float) * ALLNEIGHB * ALLNEIGHB * MAX_WIDTH);



	for (int x = 0; x < height; x++) {
		//printf("row: %d\n", x);
		//init
		now = iframe - backwardlimit;
		for (int y = 0; y < width; y++) {
			ipixels[y] = ((float)interim[x][y].r * 299 + (float)interim[x][y].g * 587 + (float)interim[x][y].b * 114 + 500) / 1000;
			ipixels[y] = log(ipixels[y] + Offset);
			//(float)interim[x][y].r;
			for (int i = backwardlimit; i <= forwardlimit; i++) {
				jpixels[i - backwardlimit][y] = log(Jpixels[i][x][y].r + Offset);
			}
			//assert(ipixels[y] > jpixels[now][y]);
		}
		//if (x == 361)
		//	printf("!!!%f\n", (float)ipixels[471]);
		//cudaMemcpy2D(gpuJpixels, pitch_Jpixels, jpixels, sizeof(float) * MAX_WIDTH, sizeof(float) * MAX_WIDTH, ALLNEIGHB, cudaMemcpyHostToDevice);
		cudaMemcpy(gpuJpixels, jpixels, sizeof(float) * ALLNEIGHB * MAX_WIDTH, cudaMemcpyHostToDevice);

		//printf("pai: %d\n", pitch_pai);
		//pai
		initpai << <1, THREAD_NUM, 0 >> >(forwardlimit - backwardlimit + 1, gpuJpixels, gpupai, width);
		//!!!
		cudaMemcpy(jpixels, gpupai, sizeof(float) * ALLNEIGHB * MAX_WIDTH, cudaMemcpyDeviceToHost);
		for (int y = 0; y < width; y++) {
			float sum = 0;
			for (int i = backwardlimit; i <= forwardlimit; i++) {
				sum += jpixels[i - backwardlimit][y];
			}
			sum = sum / (forwardlimit - backwardlimit + 1);
			sum = exp(sum) - Offset;
			DetailLayer[iframe][x][y].r = DetailLayer[iframe][x][y].g = DetailLayer[iframe][x][y].b = sum;
			DetailLayer[iframe][x][y].a = 1.0;
		}

		//PAI
		initPAI << <1, THREAD_NUM, 0 >> >(forwardlimit - backwardlimit + 1, gpuPAI, gpupai, width);
		//debug
		/*result = cudaMemcpy(jpixels, gpuPAI, sizeof(float) * ALLNEIGHB * ALLNEIGHB * MAX_WIDTH , cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) printf("Error: %s.\n", cudaGetErrorString(result));
		else {
			sprintf(name, "cache/PAI/TF-PAI-%d.txt", x);
			fp = fopen(name, "w");
			for (int i = 0; i < forwardlimit - backwardlimit + 1; i++) {
				fprintf(fp, "!!%d:\n", i);
				for (int i2 = 0; i2 < forwardlimit - backwardlimit + 1; i2++) {
					fprintf(fp, "%d:\n", i2);
					for (int y = 0; y < width; y++) {
						fprintf(fp, "%f(%d) ", jpixels[i * ALLNEIGHB + i2][y], y);
					}
					fprintf(fp, "\n");
				}
				fprintf(fp, "\n");
			}
			fclose(fp);
		}*/

		//H
		initH << <1, THREAD_NUM, 0 >> >(now, forwardlimit - backwardlimit + 1, gpuH, gpuPAI, width);
		//debug
		//result = cudaMemcpy(jpixels, gpuH, sizeof(float) * ALLNEIGHB * MAX_WIDTH , cudaMemcpyDeviceToHost);
		//if (result != cudaSuccess) printf("Error: %s.\n", cudaGetErrorString(result));
		//else {
		//	sprintf(name, "cache/TF-H-%d.txt", x);
		//	fp = fopen(name, "w");
		//	for (int i = 0; i < forwardlimit - backwardlimit + 1; i++) {
		//		for (int y = 0; y < width; y++) {
		//			fprintf(fp, "%f(%d) ", jpixels[i][y], y);
		//			//printf("%f : %d %d\n", jpixels[i][y], i, y);
		//			assert(jpixels[i][y] >= 0 && jpixels[i][y] <= 1);
		//		}
		//		fprintf(fp, "\n");
		//	}
		//	fclose(fp);
		//}

		//TF
		cudaMemcpy(gpuIpixels, ipixels, sizeof(float) * MAX_WIDTH, cudaMemcpyHostToDevice);
		gpuTemporalFiltering << <1, THREAD_NUM, sizeof(float) * MAX_WIDTH >> >
			(x, now, forwardlimit - backwardlimit + 1, gpuIpixels, gpuJpixels, gpuH, width);

		//writeback
		result = cudaMemcpy(ipixels, gpuIpixels, sizeof(float) * MAX_WIDTH, cudaMemcpyDeviceToHost);
		//sprintf(name, "cache/aTF-Ipixels-%d.txt", x);
		//fp = fopen(name, "w");
		if (result != cudaSuccess) printf("Error : %s.\n", cudaGetErrorString(result));
		else for (int y = 0; y < width; y++) {
			ipixels[y] = exp(ipixels[y]) - Offset;
			BaseLayer[iframe][x][y].r = (float)ipixels[y];
			BaseLayer[iframe][x][y].g = (float)ipixels[y];
			BaseLayer[iframe][x][y].b = (float)ipixels[y];
			//fprintf(fp, "%f(%d) ", (float)BaseLayer[iframe][x][y].r, y);
			BaseLayer[iframe][x][y].a = 1.f;
		}
		//fclose(fp);
	}

	sprintf(name, "baselayer/permeability-%03d.exr", iframe + 1);
	WriteFrame(DetailLayer[iframe][0], name);

	/*sprintf(name, "cache/ipixels%03d.txt", iframe);
	fp = fopen(name, "w");
	for (int x = 0; x < height; x++) {
	for (int y = 0; y < width; y++) {
	fprintf(fp, "%f ", (float)BaseLayer[iframe][x][y].r);
	}
	fprintf(fp, "\n");
	}
	fclose(fp);*/

	sprintf(name, "baselayer/test%03d.exr", iframe + 1);
	WriteFrame(BaseLayer[iframe][0], name);

	cudaFree(gpuIpixels);
	cudaFree(gpuJpixels);
	cudaFree(gpupai);
	cudaFree(gpuPAI);
	cudaFree(gpuH);
	return true;
}

inline
bool Frame::TemporalFiltering3channel(int iframe, int channel) {

	int backwardlimit = 0 > iframe - NEIGHBORHOOD ? 0 : iframe - NEIGHBORHOOD;
	int forwardlimit = iframe + NEIGHBORHOOD < N - 1 ? iframe + NEIGHBORHOOD : N - 1;
	int now;
	char name[NAME_SIZE];

	sprintf(name, "hallway/clip_000008.%06d.exr", iframe + 1);
	ReadFrame(iframe, name, 2);

	float *gpupai, *gpuPAI, *gpuH;
	float *gpuIpixels, *gpuJpixels;
	float ipixels[MAX_WIDTH];
	cudaError result;
	FILE *fp;

	size_t pitch_pai, pitch_PAI, pitch_H, pitch_Jpixels;
	cudaMalloc((void**)&gpuIpixels, sizeof(float) * MAX_WIDTH);

	//cudaMallocPitch((void**)&gpupai, &pitch_pai, sizeof(float) * MAX_WIDTH, ALLNEIGHB);
	//cudaMallocPitch((void**)&gpuJpixels, &pitch_Jpixels, sizeof(float) * MAX_WIDTH, ALLNEIGHB);
	cudaMalloc((void**)&gpuJpixels, sizeof(float) * ALLNEIGHB * MAX_WIDTH);
	cudaMalloc((void**)&gpupai, sizeof(float) * ALLNEIGHB * MAX_WIDTH);
	cudaMalloc((void**)&gpuH, sizeof(float) * ALLNEIGHB * MAX_WIDTH);
	//cudaMallocPitch((void**)&gpuH, &pitch_H, sizeof(float) * MAX_WIDTH, ALLNEIGHB);

	cudaMalloc((void**)&gpuPAI, sizeof(float) * ALLNEIGHB * ALLNEIGHB * MAX_WIDTH);

	Ipixels[iframe].resizeErase(height, width);

	now = iframe - backwardlimit;
	for (int x = 0; x < height; x++) {
		//printf("row: %d\n", x);
		//init
		for (int y = 0; y < width; y++) {
			switch (channel) {
			case 0: ipixels[y] = interim[x][y].r; break;
			case 1: ipixels[y] = interim[x][y].g; break;
			case 2: ipixels[y] = interim[x][y].b; break;
			}
			ipixels[y] = log(ipixels[y] + Offset);
			//(float)interim[x][y].r;
			for (int i = backwardlimit; i <= forwardlimit; i++) {
				switch (channel) {
				case 0: jpixels[i - backwardlimit][y] = log(Jpixels[i][x][y].r + Offset); break;
				case 1: jpixels[i - backwardlimit][y] = log(Jpixels[i][x][y].g + Offset); break;
				case 2: jpixels[i - backwardlimit][y] = log(Jpixels[i][x][y].b + Offset); break;
				}
				//if (i == now) printf("%f %f\n", ipixels[y], jpixels[i - backwardlimit][y]);
			}
			//assert(ipixels[y] > jpixels[now][y]);
		}
		//cudaMemcpy2D(gpuJpixels, pitch_Jpixels, jpixels, sizeof(float) * MAX_WIDTH, sizeof(float) * MAX_WIDTH, ALLNEIGHB, cudaMemcpyHostToDevice);
		result = cudaMemcpy(gpuJpixels, jpixels, sizeof(float) * ALLNEIGHB * MAX_WIDTH, cudaMemcpyHostToDevice);
		if (result != cudaSuccess) {
			printf("Error 1!\n");
		}

		//printf("pai: %d\n", pitch_pai);
		//pai
		initpai << <1, THREAD_NUM, 0 >> >(forwardlimit - backwardlimit + 1, gpuJpixels, gpupai, width);
		//debug
		/*result = cudaMemcpy(jpixels, gpupai, sizeof(float) * ALLNEIGHB * MAX_WIDTH, cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) printf("Error: %s.\n", cudaGetErrorString(result));
		else {
		sprintf(name, "cache/gpupai/TF-PAI-%d.txt", x);
		fp = fopen(name, "w");
		for (int i = 0; i < forwardlimit - backwardlimit + 1; i++) {
		for (int y = 0; y < width; y++) {
		fprintf(fp, "%f(%d) ", jpixels[i][y], y);
		}
		fprintf(fp, "\n");
		}
		fclose(fp);
		}*/

		//PAI
		initPAI << <1, THREAD_NUM, 0 >> >(forwardlimit - backwardlimit + 1, gpuPAI, gpupai, width);
		//debug
		/*result = cudaMemcpy(jpixels, gpuPAI, sizeof(float) * ALLNEIGHB * ALLNEIGHB * MAX_WIDTH , cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) printf("Error: %s.\n", cudaGetErrorString(result));
		else {
		sprintf(name, "cache/PAI/TF-PAI-%d.txt", x);
		fp = fopen(name, "w");
		for (int i = 0; i < forwardlimit - backwardlimit + 1; i++) {
		fprintf(fp, "!!%d:\n", i);
		for (int i2 = 0; i2 < forwardlimit - backwardlimit + 1; i2++) {
		fprintf(fp, "%d:\n", i2);
		for (int y = 0; y < width; y++) {
		fprintf(fp, "%f(%d) ", jpixels[i * ALLNEIGHB + i2][y], y);
		}
		fprintf(fp, "\n");
		}
		fprintf(fp, "\n");
		}
		fclose(fp);
		}*/

		//H
		initH << <1, THREAD_NUM, 0 >> >(now, forwardlimit - backwardlimit + 1, gpuH, gpuPAI, width);
		//debug
		//result = cudaMemcpy(jpixels, gpuH, sizeof(float) * ALLNEIGHB * MAX_WIDTH , cudaMemcpyDeviceToHost);
		//if (result != cudaSuccess) printf("Error: %s.\n", cudaGetErrorString(result));
		//else {
		//	sprintf(name, "cache/TF-H-%d.txt", x);
		//	fp = fopen(name, "w");
		//	for (int i = 0; i < forwardlimit - backwardlimit + 1; i++) {
		//		for (int y = 0; y < width; y++) {
		//			fprintf(fp, "%f(%d) ", jpixels[i][y], y);
		//			//printf("%f : %d %d\n", jpixels[i][y], i, y);
		//			assert(jpixels[i][y] >= 0 && jpixels[i][y] <= 1);
		//		}
		//		fprintf(fp, "\n");
		//	}
		//	fclose(fp);
		//}

		//TF
		cudaMemcpy(gpuIpixels, ipixels, sizeof(float) * MAX_WIDTH, cudaMemcpyHostToDevice);
		gpuTemporalFiltering << <1, THREAD_NUM, sizeof(float) * MAX_WIDTH >> >
			(x, now, forwardlimit - backwardlimit + 1, gpuIpixels, gpuJpixels, gpuH, width);

		//writeback
		result = cudaMemcpy(ipixels, gpuIpixels, sizeof(float) * MAX_WIDTH, cudaMemcpyDeviceToHost);
		//sprintf(name, "cache/channel/TF-Ipixels-%d.txt", x);
		//fp = fopen(name, "w");
		if (result != cudaSuccess) printf("Error : %s.\n", cudaGetErrorString(result));
		else for (int y = 0; y < width; y++) {
			switch (channel) {
			case 0:
				BaseLayer[iframe][x][y].r = log(BaseLayer[iframe][x][y].r + Offset);
				DetailLayer[iframe][x][y].r = (float)ipixels[y];
				BaseLayer[iframe][x][y].r = DetailLayer[iframe][x][y].r + BaseLayer[iframe][x][y].r * Sigma_r;

				DetailLayer[iframe][x][y].r = exp(DetailLayer[iframe][x][y].r) - Offset;
				BaseLayer[iframe][x][y].r = exp(BaseLayer[iframe][x][y].r) - Offset;
				img.at<float>(x, y * img.channels() + 2) = (float)BaseLayer[iframe][x][y].r * 10;
				//fprintf(fp, "%f(%d) ", (float)DetailLayer[iframe][x][y].r, y);
				break;
			case 1:
				BaseLayer[iframe][x][y].g = log(BaseLayer[iframe][x][y].g + Offset);
				DetailLayer[iframe][x][y].g = (float)ipixels[y];
				BaseLayer[iframe][x][y].g = DetailLayer[iframe][x][y].g + BaseLayer[iframe][x][y].g * Sigma_r;

				DetailLayer[iframe][x][y].g = exp(DetailLayer[iframe][x][y].g) - Offset;
				BaseLayer[iframe][x][y].g = exp(BaseLayer[iframe][x][y].g) - Offset;
				img.at<float>(x, y * img.channels() + 1) = (float)BaseLayer[iframe][x][y].g * 10;
				//fprintf(fp, "%f(%d) ", (float)DetailLayer[iframe][x][y].g, y);
				break;
			case 2:
				BaseLayer[iframe][x][y].b = log(BaseLayer[iframe][x][y].b + Offset);
				DetailLayer[iframe][x][y].b = (float)ipixels[y];
				BaseLayer[iframe][x][y].b = DetailLayer[iframe][x][y].b + BaseLayer[iframe][x][y].b * Sigma_r;

				DetailLayer[iframe][x][y].b = exp(DetailLayer[iframe][x][y].b) - Offset;
				BaseLayer[iframe][x][y].b = exp(BaseLayer[iframe][x][y].b) - Offset;
				img.at<float>(x, y * img.channels() + 0) = (float)BaseLayer[iframe][x][y].b * 10;
				//fprintf(fp, "%f(%d) ", (float)DetailLayer[iframe][x][y].b, y);
				break;
			}
			//fprintf(fp, "%f,%f,%f\n", (float)BaseLayer[iframe][x][y].r, (float)BaseLayer[iframe][x][y].g, (float)BaseLayer[iframe][x][y].b);
			DetailLayer[iframe][x][y].a = 1.f;
		}
		//fclose(fp);
	}

	/*sprintf(name, "cache/ipixels%03d.txt", iframe);
	fp = fopen(name, "w");
	for (int x = 0; x < height; x++) {
	for (int y = 0; y < width; y++) {
	fprintf(fp, "%f ", (float)BaseLayer[iframe][x][y].r);
	}
	fprintf(fp, "\n");
	}
	fclose(fp);*/
	if (channel == 2) {
		sprintf(name, "detaillayer/timecrazy%03d.exr", iframe + 1);
		WriteFrame(DetailLayer[iframe][0], name);
		//sprintf(name, "tonemapped/1crazy%03d.exr", iframe + 1);
		//WriteFrame(BaseLayer[iframe][0], name);
		//sprintf(name, "tonemapped/1crazy%03d.png", iframe + 1);
		//cvSaveImage(name, &(IplImage(img)));
		//writer << img;
	}

	cudaFree(gpuIpixels);
	cudaFree(gpuJpixels);
	cudaFree(gpupai);
	cudaFree(gpuPAI);
	cudaFree(gpuH);
	return true;
}

#endif //TEMPORALFILTERING_H_INCLUDE