#ifndef TM_H_INCLUDE
#define TM_H_INCLUDE

#include <iostream>
#include <cstring>
#include <cstdio>

//cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//opencv
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

//openexr
#include <half.h>
#include <ImfRgba.h>
#include <ImfArray.h>
#include <ImfRgbaFile.h>
#include <ImfStringAttribute.h>
#include <ImfMatrixAttribute.h>

using namespace std;
using namespace cv;
using namespace Imf;
using namespace Imath;

#define THREAD_NUM 512

#define MAX_FRAME 332
#define MAX_HEIGHT 760
#define MAX_WIDTH 1300
#define NAME_SIZE 64
#define Lambda (float)1
#define Sigma 0.1
#define PHOTO_CONSTANCY_Sigma 0.1
#define FLOW_GRADIENT_Sigma 2
#define Sigma_r 0.4
#define Offset 0.00001
#define NEIGHBORHOOD 10

float pixels[MAX_HEIGHT][MAX_WIDTH];

class Frame {
public:
	int N;
	int width, height;
	bool ReadFrame(int i, char filename[], const int type);
	bool ReadFrames(char filename[]);
	bool WriteFrame(const Rgba *Pixels, const char filename[]);
	bool WriteFrames();
	bool SpatialFiltering(int i);
	bool AllSpatialFiltering(int start, int end);

	bool TemporalFiltering3channel(int iframe, int channel);
	bool AllTemporalFiltering3channel();
	bool Warping3channel(int frame, int channel);

	bool TemporalFiltering(int iframe);
	bool AllTemporalFiltering();
	bool Warping(int iframe);
private:
	Mat pic[MAX_FRAME];
	Array2D<Rgba> Ipixels[MAX_FRAME], Jpixels[MAX_FRAME], interim;
	Array2D<Rgba> BaseLayer[MAX_FRAME], DetailLayer[MAX_FRAME];
	char strFilename[NAME_SIZE];
	Mat img;
	VideoWriter writer;
};

bool Frame::WriteFrame(const Rgba *Pixels, const char filename[]) {
	RgbaOutputFile file(filename, width, height, WRITE_RGBA); // 1
	file.setFrameBuffer(Pixels, 1, width); // 2
	file.writePixels(height); // 3
	return true;
}

bool Frame::WriteFrames() {
	char name[NAME_SIZE];
	for (int i = 1; i <= N; i++) {
		sprintf(name, "%s%06d-after.exr", strFilename, i);
		if (!WriteFrame(Ipixels[i][0], name)) return false;
	}
	return true;
}

bool Frame::ReadFrame(int i, char filename[], const int type) {
	//printf("%s %d\n", filename, i);
	RgbaInputFile file(filename);
	Box2i dw = file.dataWindow();
	width = dw.max.x - dw.min.x + 1;
	height = dw.max.y - dw.min.y + 1;
	if (type == 1) {
		Ipixels[i].resizeErase(height, width);
		file.setFrameBuffer(&Ipixels[i][0][0] - dw.min.x - dw.min.y * width, 1, width);
	}
	else if (type == 0) {
		BaseLayer[i].resizeErase(height, width);
		file.setFrameBuffer(&BaseLayer[i][0][0] - dw.min.x - dw.min.y * width, 1, width);
	}
	else if (type == 2) {
		interim.resizeErase(height, width);
		file.setFrameBuffer(&interim[0][0] - dw.min.x - dw.min.y * width, 1, width);
	}
	file.readPixels(dw.min.y, dw.max.y);
	return true;
}

bool Frame::ReadFrames(char filename[]) {
	char name[NAME_SIZE];
	strcpy(strFilename, filename);
	for (int i = 1; i <= N; i++) {
		sprintf(name, "%s%06d.exr", strFilename, i);
		if (!ReadFrame(i - 1, name, 1)) return false;
	}
	return true;
}

bool Frame::AllSpatialFiltering(int start, int end) {
	char name[NAME_SIZE];

	/*SpatialFiltering(1);
	sprintf(name, "SF-0.1-%03d-2.exr", start);
	WriteFrame(BaseLayer[start][0], name);

	return 0;*/
	for (int i = start - 1; i < end; i++) {
		printf("Frame No.%03d:\n", i);
		SpatialFiltering(i);
		sprintf(name, "sf/0.1-%03d-hallway.exr", i + 1);
		WriteFrame(BaseLayer[i][0], name);
	}
	return true;
}

inline
bool Frame::AllTemporalFiltering() {
	char name[NAME_SIZE];
	float max, min;
	//read all frame to Ipixels
	for (int i = 0; i < 22; i++) {
		printf("pre: %d\n", i);
		//sprintf(name, "result/spatialfiler%03d-10-300.exr", i + 1);
		sprintf(name, "sf/0.1-%03d-hallway.exr", i + 1);
		ReadFrame(i, name, 1);

		//sprintf(name, "result/spatialfiler%03d-2-300.png", i+1);
		//WriteFrame(Ipixels[i][0], name);

		pic[i] = Mat(height, width, CV_MAKETYPE(CV_32F, 1)); //cv32f->float, 1->single channel

		max = 0.0; min = 0.0;
		for (int x = 0; x < height; x++) {
			for (int y = 0; y < width; y++) {
				if ((float)Ipixels[i][x][y].r > max) max = (float)Ipixels[i][x][y].r;
				if ((float)Ipixels[i][x][y].r < min) min = (float)Ipixels[i][x][y].r;
			}
		}

		for (int x = 0; x < height; x++) {
			for (int y = 0; y < width; y++) {
				pic[i].at<float>(x, y) = log((float)Ipixels[i][x][y].r) * 255.f / log(max + 1);
				//printf("%f\n", pic[i].at<float>(x, y));
				assert(pic[i].at<float>(x, y) >= 0 && pic[i].at<float>(x, y) <= 255);
			}
		}
	}

	for (int i = 10; i < 11; i++) {
		printf("Frame : %d\n", i);
		Warping(i);
		TemporalFiltering(i);
	}
	return true;
}

inline
bool Frame::AllTemporalFiltering3channel() {
	char name[NAME_SIZE];
	float max, min, luminance;
	//read all frame to Ipixels

	for (int i = 0; i < 22; i++) {
		printf("pre: %d\n", i);
		sprintf(name, "hallway/clip_000008.%06d.exr", i + 1);
		ReadFrame(i, name, 1); // read input HDR frame to Ipixels

		pic[i] = Mat(height, width, CV_MAKETYPE(CV_32F, 1)); //cv32f->float, 1->single channel

		max = 0.0; min = 0.0;
		for (int x = 0; x < height; x++) {
			for (int y = 0; y < width; y++) {
				luminance = ((float)Ipixels[i][x][y].r * 299 + (float)Ipixels[i][x][y].g * 587 + (float)Ipixels[i][x][y].b * 114 + 500) / 1000;
				if ((float)luminance > max) max = (float)luminance;
				if ((float)luminance < min) min = (float)luminance;
			}
		}

		for (int x = 0; x < height; x++) {
			for (int y = 0; y < width; y++) {
				luminance = ((float)Ipixels[i][x][y].r * 299 + (float)Ipixels[i][x][y].g * 587 + (float)Ipixels[i][x][y].b * 114 + 500) / 1000;
				pic[i].at<float>(x, y) = log((float)luminance) * 255.f / log(max + 1);
				//printf("%f\n", pic[i].at<float>(x, y));
				assert(pic[i].at<float>(x, y) >= 0 && pic[i].at<float>(x, y) <= 255);
			}
		}
	}

	//writer = VideoWriter("tonemapped/VideoTest.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10.0, Size(width, height));
	for (int i = 10; i < 11; i++) {
		printf("Frame : %d\n", i);

		sprintf(name, "baselayer/test%03d.exr", i + 1);
		ReadFrame(i, name, 0);

		Warping3channel(i, 0);
		Warping3channel(i, 1);
		Warping3channel(i, 2);

		img = Mat(height, width, CV_MAKETYPE(CV_32F, 3));
		DetailLayer[i].resizeErase(height, width);
		TemporalFiltering3channel(i, 0);
		TemporalFiltering3channel(i, 1);
		TemporalFiltering3channel(i, 2);
	}
	return true;
}

#endif //TM_H_INCLUDE