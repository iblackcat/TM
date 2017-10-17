#ifndef WARPING_H_INCLUDE
#define WARPING_H_INCLUDE

#include "tm.cuh"

#define UNKNOWN_FLOW_THRESH 1e9

float change_x[MAX_HEIGHT][MAX_WIDTH];
float change_y[MAX_HEIGHT][MAX_WIDTH];
float cache_x[MAX_HEIGHT][MAX_WIDTH], cache_y[MAX_HEIGHT][MAX_WIDTH];

void makecolorwheel(vector<Scalar> &colorwheel)
{
	int RY = 15;
	int YG = 6;
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;

	int i;

	for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255, 255 * i / RY, 0));
	for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));
	for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0, 255, 255 * i / GC));
	for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));
	for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255 * i / BM, 0, 255));
	for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));
}

void motionToColor(Mat flow, Mat &color)
{
	if (color.empty())
		color.create(flow.rows, flow.cols, CV_8UC3);

	static vector<Scalar> colorwheel; //Scalar r,g,b
	if (colorwheel.empty())
		makecolorwheel(colorwheel);

	// determine motion range:
	float maxrad = -1;

	// Find max flow to normalize fx and fy
	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);
			float fx = flow_at_point[0];
			float fy = flow_at_point[1];
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
				continue;
			float rad = sqrt(fx * fx + fy * fy);
			maxrad = maxrad > rad ? maxrad : rad;
		}
	}

	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			uchar *data = color.data + color.step[0] * i + color.step[1] * j;
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);

			float fx = flow_at_point[0] / maxrad;
			float fy = flow_at_point[1] / maxrad;
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
			{
				data[0] = data[1] = data[2] = 0;
				continue;
			}
			float rad = sqrt(fx * fx + fy * fy);

			float angle = atan2(-fy, -fx) / CV_PI;
			float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);
			int k0 = (int)fk;
			int k1 = (k0 + 1) % colorwheel.size();
			float f = fk - k0;
			//f = 0; // uncomment to see original color wheel

			for (int b = 0; b < 3; b++)
			{
				float col0 = colorwheel[k0][b] / 255.0;
				float col1 = colorwheel[k1][b] / 255.0;
				float col = (1 - f) * col0 + f * col1;
				if (rad <= 1)
					col = 1 - rad * (1 - col); // increase saturation with radius
				else
					col *= .75; // out of range
				data[2 - b] = (int)(255.0 * col);
			}
		}
	}
}

__global__
void gpuWarping(int row, float *gpuIpixels, int Ipixels_width, float *gpuJpixels, int Jpixels_width, float *gpu_x, int x_width, float *gpu_y, int y_width, int Height, int Width) {
	const int tid = threadIdx.x;
	int x1, y1, x2, y2, x, y;
	float wx, wy;
	for (int i = tid; i < Width; i += THREAD_NUM) {
		//平均四块
		/*wx = gpu_x[row * x_width + i] - floor(gpu_x[row * x_width + i]);
		wy = gpu_y[row * y_width + i] - floor(gpu_y[row * y_width + i]);
		x1 = row + floor(gpu_x[row * x_width + i]);
		y1 = i + floor(gpu_y[row * y_width + i]);
		x2 = x1 + 1;
		y2 = y1 + 1;

		if (x1 < 0) x1 = 0; else if (x1 >= Height) x1 = Height - 1;
		if (x2 < 0) x2 = 0; else if (x2 >= Height) x2 = Height - 1;
		if (y1 < 0) y1 = 0; else if (y1 >= Width) y1 = Width - 1;
		if (y2 < 0) y2 = 0; else if (y2 >= Width) y2 = Width - 1;

		gpuJpixels[row * Jpixels_width + i] = (float)gpuIpixels[x1 * Ipixels_width + y1] * wx * wy
		+ (float)gpuIpixels[x1 * Ipixels_width + y2] * wx * (1.0 - wy)
		+ (float)gpuIpixels[x2 * Ipixels_width + y1] * (1.0 - wx) * wy
		+ (float)gpuIpixels[x2 * Ipixels_width + y2] * (1.0 - wx) * (1.0 - wy);*/

		//四舍五入
		x = row + floor(gpu_x[row * x_width + i] + 0.5);
		y = i + floor(gpu_y[row * y_width + i] + 0.5);
		if (x < 0) x = 0;
		else if (x >= Height) x = Height - 1;
		if (y < 0) y = 0;
		else if (y >= Width) y = Width - 1;
		gpuJpixels[row * Jpixels_width + i] = gpuIpixels[x * Ipixels_width + y];
	}
}



inline
bool Frame::Warping(int iframe) {
	char name[NAME_SIZE];
	float *gpuIpixels, *gpuJpixels, *gpu_x, *gpu_y;
	size_t pitch_Px, pitch_Py, pitch_Ipixels, pitch_Jpixels;
	FILE *fp;

	cudaMallocPitch((void**)&gpu_x, &pitch_Px, sizeof(float) * MAX_WIDTH, MAX_HEIGHT);
	cudaMallocPitch((void**)&gpu_y, &pitch_Py, sizeof(float) * MAX_WIDTH, MAX_HEIGHT);
	cudaMallocPitch((void**)&gpuIpixels, &pitch_Ipixels, sizeof(float) * MAX_WIDTH, MAX_HEIGHT);
	cudaMallocPitch((void**)&gpuJpixels, &pitch_Jpixels, sizeof(float) * MAX_WIDTH, MAX_HEIGHT);

	int backwardlimit = 0 > iframe - NEIGHBORHOOD ? 0 : iframe - NEIGHBORHOOD;
	int forwardlimit = iframe + NEIGHBORHOOD < N - 1 ? iframe + NEIGHBORHOOD : N - 1;
	printf("backwardlimit: %d, forwardlimit: %d\n", backwardlimit, forwardlimit);

	Mat flow;//	Mat prevgray, gray, flow, cflow, frame;

	//calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
	//flow.at<vec2f>(i,j)[0] means x component of the flow at the position (i,j); flow.at<vec2f>(i,j)[1] means y component of the flow at the position (i,j);

	memset(change_x, 0, sizeof(change_x));
	memset(change_y, 0, sizeof(change_y));

	Jpixels[iframe].resizeErase(height, width);
	//fp = fopen("out/aaaaa.txt", "w");
	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
			cache_x[x][y] = x;
			cache_y[x][y] = y;

			Jpixels[iframe][x][y].r = Jpixels[iframe][x][y].g = Jpixels[iframe][x][y].b = Ipixels[iframe][x][y].r;
			Jpixels[iframe][x][y].a = Ipixels[iframe][x][y].a;
			//fprintf(fp, "%f(%d) ", (float)Jpixels[iframe][x][y].r, y);
		}
		//fprintf(fp, "\n");
	}
	//fclose(fp);

	int prev = iframe, nx, ny;
	//cvNamedWindow("prev");
	//cvNamedWindow("now");

	for (int i = iframe - 1; i >= backwardlimit; i--) {
		printf("Now : %d, Prev : %d\n", i, prev);

		//imshow("prev", pic[prev]);
		//cvSaveImage("test.png", &(IplImage(pic[prev])));
		//imwrite("a.png", pic[prev]);
		//imshow("now", pic[i]);

		calcOpticalFlowFarneback(pic[prev], pic[i], flow, 0.5, 3, 15, 3, 5, 1.2, 0);


		for (int x = 0; x < height; x++) {
			for (int y = 0; y < width; y++) {
				//init last iteration location x
				if (x + cache_x[x][y] < 0) nx = 0;
				else if (x + cache_x[x][y] >= height) nx = height - 1;
				else nx = x + cache_x[x][y];

				//init last iteration location y
				if (y + cache_y[x][y] < 0) ny = 0;
				else if (y + cache_y[x][y] >= width) ny = width - 1;
				else ny = y + cache_y[x][y];

				change_x[x][y] += flow.at<Vec2f>(nx, ny)[0];
				change_y[x][y] += flow.at<Vec2f>(nx, ny)[1];

				pixels[x][y] = (float)Ipixels[iframe][x][y].r;
			}
		}

		cudaMemcpy2D(gpu_x, pitch_Px, change_x, sizeof(float) * MAX_WIDTH, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyHostToDevice);
		cudaMemcpy2D(gpu_y, pitch_Py, change_y, sizeof(float) * MAX_WIDTH, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyHostToDevice);
		cudaMemcpy2D(gpuIpixels, pitch_Ipixels, pixels, sizeof(float) * MAX_WIDTH, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyHostToDevice);

		for (int x = 0; x < height; x++) {
			gpuWarping << <1, THREAD_NUM, 0 >> >
				(x, gpuIpixels, pitch_Ipixels / sizeof(float), gpuJpixels, pitch_Jpixels / sizeof(float), gpu_x, pitch_Px / sizeof(float), gpu_y, pitch_Py / sizeof(float), height, width);
		}

		cudaMemcpy2D(pixels, sizeof(float) * MAX_WIDTH, gpuJpixels, pitch_Jpixels, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyDeviceToHost);
		Jpixels[i].resizeErase(height, width);
		for (int x = 0; x < height; x++) {
			for (int y = 0; y < width; y++) {
				Jpixels[i][x][y].r = Jpixels[i][x][y].g = Jpixels[i][x][y].b = pixels[x][y];
				Jpixels[i][x][y].a = Ipixels[i][x][y].a;

				cache_x[x][y] = change_x[x][y];
				cache_y[x][y] = change_y[x][y];
			}
		}
		sprintf(name, "out/test%03d.exr", i + 1);
		WriteFrame(Jpixels[i][0], name);
		prev = i;
	}


	memset(change_x, 0, sizeof(change_x));
	memset(change_y, 0, sizeof(change_y));

	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
			cache_x[x][y] = x;
			cache_y[x][y] = y;
		}
	}

	prev = iframe;
	for (int i = iframe + 1; i <= forwardlimit; i++) {
		printf("Now : %d, Prev : %d\n", i, prev);

		//imshow("prev", pic[prev]);
		//cvSaveImage("test.png", &(IplImage(pic[prev])));
		//imwrite("a.png", pic[prev]);
		//imshow("now", pic[i]);

		calcOpticalFlowFarneback(pic[prev], pic[i], flow, 0.5, 3, 15, 3, 5, 1.2, 0);

		for (int x = 0; x < height; x++) {
			for (int y = 0; y < width; y++) {
				//init last iteration location x
				if (x + cache_x[x][y] < 0) nx = 0;
				else if (x + cache_x[x][y] >= height) nx = height - 1;
				else nx = x + cache_x[x][y];

				//init last iteration location y
				if (y + cache_y[x][y] < 0) ny = 0;
				else if (y + cache_y[x][y] >= width) ny = width - 1;
				else ny = y + cache_y[x][y];

				change_x[x][y] += flow.at<Vec2f>(nx, ny)[1];
				change_y[x][y] += flow.at<Vec2f>(nx, ny)[0];

				pixels[x][y] = Ipixels[iframe][x][y].r;
			}
		}

		cudaMemcpy2D(gpu_x, pitch_Px, change_x, sizeof(float) * MAX_WIDTH, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyHostToDevice);
		cudaMemcpy2D(gpu_y, pitch_Py, change_y, sizeof(float) * MAX_WIDTH, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyHostToDevice);
		cudaMemcpy2D(gpuIpixels, pitch_Ipixels, pixels, sizeof(float) * MAX_WIDTH, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyHostToDevice);

		for (int x = 0; x < height; x++) {
			gpuWarping << <1, THREAD_NUM, 0 >> >
				(x, gpuIpixels, pitch_Ipixels / sizeof(float), gpuJpixels, pitch_Jpixels / sizeof(float), gpu_x, pitch_Px / sizeof(float), gpu_y, pitch_Py / sizeof(float), height, width);
		}

		cudaMemcpy2D(pixels, sizeof(float) * MAX_WIDTH, gpuJpixels, pitch_Jpixels, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyDeviceToHost);
		Jpixels[i].resizeErase(height, width);

		//sprintf(name, "out/pixels%03d.txt", i);
		//fp = fopen(name, "w");
		for (int x = 0; x < height; x++) {
			for (int y = 0; y < width; y++) {
				Jpixels[i][x][y].r = Jpixels[i][x][y].g = Jpixels[i][x][y].b = (float)pixels[x][y];
				Jpixels[i][x][y].a = Ipixels[i][x][y].a;
				//fprintf(fp, "%f(%d) ", (float)Jpixels[i][x][y].r, y);

				cache_x[x][y] = change_x[x][y];
				cache_y[x][y] = change_y[x][y];
			}
			//fprintf(fp, "\n");
		}
		//fclose(fp);

		sprintf(name, "out/test%03d.exr", i + 1);
		WriteFrame(Jpixels[i][0], name);
		prev = i;
	}

	cudaFree(gpu_x);
	cudaFree(gpu_y);
	cudaFree(gpuIpixels);
	cudaFree(gpuJpixels);
	return true;
}

inline
bool Frame::Warping3channel(int iframe, int channel) {
	char name[NAME_SIZE];
	float *gpuIpixels, *gpuJpixels, *gpu_x, *gpu_y;
	size_t pitch_Px, pitch_Py, pitch_Ipixels, pitch_Jpixels;
	FILE *fp;

	cudaMallocPitch((void**)&gpu_x, &pitch_Px, sizeof(float) * MAX_WIDTH, MAX_HEIGHT);
	cudaMallocPitch((void**)&gpu_y, &pitch_Py, sizeof(float) * MAX_WIDTH, MAX_HEIGHT);
	cudaMallocPitch((void**)&gpuIpixels, &pitch_Ipixels, sizeof(float) * MAX_WIDTH, MAX_HEIGHT);
	cudaMallocPitch((void**)&gpuJpixels, &pitch_Jpixels, sizeof(float) * MAX_WIDTH, MAX_HEIGHT);

	int backwardlimit = 0 > iframe - NEIGHBORHOOD ? 0 : iframe - NEIGHBORHOOD;
	int forwardlimit = iframe + NEIGHBORHOOD < N - 1 ? iframe + NEIGHBORHOOD : N - 1;
	printf("backwardlimit: %d, forwardlimit: %d\n", backwardlimit, forwardlimit);

	Mat flow;//	Mat prevgray, gray, flow, cflow, frame;

	//calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
	//flow.at<vec2f>(i,j)[0] means x component of the flow at the position (i,j); flow.at<vec2f>(i,j)[1] means y component of the flow at the position (i,j);

	memset(change_x, 0, sizeof(change_x));
	memset(change_y, 0, sizeof(change_y));

	Jpixels[iframe].resizeErase(height, width);
	//fp = fopen("out/aaaaa.txt", "w");
	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
			cache_x[x][y] = x;
			cache_y[x][y] = y;

			Jpixels[iframe][x][y].r = Ipixels[iframe][x][y].r;
			Jpixels[iframe][x][y].g = Ipixels[iframe][x][y].g;
			Jpixels[iframe][x][y].b = Ipixels[iframe][x][y].b;
			Jpixels[iframe][x][y].a = Ipixels[iframe][x][y].a;
			//fprintf(fp, "%f(%d) ", (float)Jpixels[iframe][x][y].r, y);
		}
		//fprintf(fp, "\n");
	}
	//fclose(fp);

	int prev = iframe, nx, ny;
	//cvNamedWindow("prev");
	//cvNamedWindow("now");

	for (int i = iframe - 1; i >= backwardlimit; i--) {
		printf("Now : %d, Prev : %d\n", i, prev);

		//imshow("prev", pic[prev]);
		//cvSaveImage("test.png", &(IplImage(pic[prev])));
		//imwrite("a.png", pic[prev]);
		//imshow("now", pic[i]);

		calcOpticalFlowFarneback(pic[prev], pic[i], flow, 0.5, 3, 15, 3, 5, 1.2, 0);


		for (int x = 0; x < height; x++) {
			for (int y = 0; y < width; y++) {
				//init last iteration location x
				if (x + cache_x[x][y] < 0) nx = 0;
				else if (x + cache_x[x][y] >= height) nx = height - 1;
				else nx = x + cache_x[x][y];

				//init last iteration location y
				if (y + cache_y[x][y] < 0) ny = 0;
				else if (y + cache_y[x][y] >= width) ny = width - 1;
				else ny = y + cache_y[x][y];

				change_x[x][y] += flow.at<Vec2f>(nx, ny)[1];
				change_y[x][y] += flow.at<Vec2f>(nx, ny)[0];

				switch (channel) {
				case 0: pixels[x][y] = (float)Ipixels[iframe][x][y].r; break;
				case 1: pixels[x][y] = (float)Ipixels[iframe][x][y].g; break;
				case 2: pixels[x][y] = (float)Ipixels[iframe][x][y].b; break;
				}
			}
		}

		cudaMemcpy2D(gpu_x, pitch_Px, change_x, sizeof(float) * MAX_WIDTH, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyHostToDevice);
		cudaMemcpy2D(gpu_y, pitch_Py, change_y, sizeof(float) * MAX_WIDTH, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyHostToDevice);
		cudaMemcpy2D(gpuIpixels, pitch_Ipixels, pixels, sizeof(float) * MAX_WIDTH, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyHostToDevice);

		for (int x = 0; x < height; x++) {
			gpuWarping << <1, THREAD_NUM, 0 >> >
				(x, gpuIpixels, pitch_Ipixels / sizeof(float), gpuJpixels, pitch_Jpixels / sizeof(float), gpu_x, pitch_Px / sizeof(float), gpu_y, pitch_Py / sizeof(float), height, width);
		}

		cudaMemcpy2D(pixels, sizeof(float) * MAX_WIDTH, gpuJpixels, pitch_Jpixels, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyDeviceToHost);
		if (channel == 0)
			Jpixels[i].resizeErase(height, width);

		//sprintf(name, "out/3channel-pixels%03d.txt", i);
		//fp = fopen(name, "w");
		for (int x = 0; x < height; x++) {
			for (int y = 0; y < width; y++) {
				switch (channel) {
				case 0: Jpixels[i][x][y].r = pixels[x][y]; break;
				case 1: Jpixels[i][x][y].g = pixels[x][y]; break;
				case 2: Jpixels[i][x][y].b = pixels[x][y]; break;
				}
				Jpixels[i][x][y].a = 1.f;
				//fprintf(fp, "%f,%f,%f(%d) ", (float)Jpixels[i][x][y].r, (float)Jpixels[i][x][y].g, (float)Jpixels[i][x][y].b, y);

				cache_x[x][y] = change_x[x][y];
				cache_y[x][y] = change_y[x][y];
			}
			//fprintf(fp, "\n");
		}
		//fclose(fp);
		if (channel == 2) {
		sprintf(name, "out/3channel-%03d.exr", i + 1);
		WriteFrame(Jpixels[i][0], name);
		}
		prev = i;
	}


	memset(change_x, 0, sizeof(change_x));
	memset(change_y, 0, sizeof(change_y));

	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
			cache_x[x][y] = x;
			cache_y[x][y] = y;
		}
	}

	prev = iframe;
	for (int i = iframe + 1; i <= forwardlimit; i++) {
		printf("Now : %d, Prev : %d\n", i, prev);

		//imshow("prev", pic[prev]);
		//cvSaveImage("test.png", &(IplImage(pic[prev])));
		//imwrite("a.png", pic[prev]);
		//imshow("now", pic[i]);

		calcOpticalFlowFarneback(pic[prev], pic[i], flow, 0.5, 3, 15, 3, 5, 1.2, 0);


		for (int x = 0; x < height; x++) {
			for (int y = 0; y < width; y++) {
				//init last iteration location x
				if (x + cache_x[x][y] < 0) nx = 0;
				else if (x + cache_x[x][y] >= height) nx = height - 1;
				else nx = x + cache_x[x][y];

				//init last iteration location y
				if (y + cache_y[x][y] < 0) ny = 0;
				else if (y + cache_y[x][y] >= width) ny = width - 1;
				else ny = y + cache_y[x][y];

				change_x[x][y] += flow.at<Vec2f>(nx, ny)[1];
				change_y[x][y] += flow.at<Vec2f>(nx, ny)[0];

				switch (channel) {
				case 0: pixels[x][y] = (float)Ipixels[iframe][x][y].r; break;
				case 1: pixels[x][y] = (float)Ipixels[iframe][x][y].g; break;
				case 2: pixels[x][y] = (float)Ipixels[iframe][x][y].b; break;
				}
			}
		}

		cudaMemcpy2D(gpu_x, pitch_Px, change_x, sizeof(float) * MAX_WIDTH, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyHostToDevice);
		cudaMemcpy2D(gpu_y, pitch_Py, change_y, sizeof(float) * MAX_WIDTH, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyHostToDevice);
		cudaMemcpy2D(gpuIpixels, pitch_Ipixels, pixels, sizeof(float) * MAX_WIDTH, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyHostToDevice);

		for (int x = 0; x < height; x++) {
			gpuWarping << <1, THREAD_NUM, 0 >> >
				(x, gpuIpixels, pitch_Ipixels / sizeof(float), gpuJpixels, pitch_Jpixels / sizeof(float), gpu_x, pitch_Px / sizeof(float), gpu_y, pitch_Py / sizeof(float), height, width);
		}

		cudaMemcpy2D(pixels, sizeof(float) * MAX_WIDTH, gpuJpixels, pitch_Jpixels, sizeof(float) * MAX_WIDTH, MAX_HEIGHT, cudaMemcpyDeviceToHost);
		if (channel == 0)
			Jpixels[i].resizeErase(height, width);

		//sprintf(name, "out/3channel-pixels%03d.txt", i);
		//fp = fopen(name, "w");
		for (int x = 0; x < height; x++) {
			for (int y = 0; y < width; y++) {
				switch (channel) {
				case 0: Jpixels[i][x][y].r = pixels[x][y]; break;
				case 1: Jpixels[i][x][y].g = pixels[x][y]; break;
				case 2: Jpixels[i][x][y].b = pixels[x][y]; break;
				}
				Jpixels[i][x][y].a = 1.f;
				//fprintf(fp, "%f(%d) ", (float)Jpixels[i][x][y].r, y);

				cache_x[x][y] = change_x[x][y];
				cache_y[x][y] = change_y[x][y];
			}
			//fprintf(fp, "\n");
		}
		//fclose(fp);
		if (channel == 2) {
		sprintf(name, "out/3channel-%03d.exr", i + 1);
		WriteFrame(Jpixels[i][0], name);
		}
		prev = i;
	}

	cudaFree(gpu_x);
	cudaFree(gpu_y);
	cudaFree(gpuIpixels);
	cudaFree(gpuJpixels);
	return true;
}

#endif //WARPING_H_INCLUDE