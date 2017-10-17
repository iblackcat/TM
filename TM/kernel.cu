#include "tm.cuh"
#include "Warping.cuh"
#include "SpatialFiltering.cuh"
#include "TemporalFiltering.cuh"

bool InitCUDA() {
	int count;
	cudaGetDeviceCount(&count);
	if (!count) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	int i;
	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}

	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);
	return true;
}

int main() {
	if (!InitCUDA()) {
		return 0;
	}

	/*Mat img = imread("img.jpg");
	namedWindow("img");
	imshow("img", img);
	waitKey(6000);*/

	Frame pic;

	int start = 1, end = 24;
	pic.N = 25;
	//pic.ReadFrame(1,"hallway/clip_000008.000001.exr",1);
	//pic.ReadFrames("hallway/clip_000008.");
	//printf("%d\n%d\n", pic.height, pic.width);
	//pic.AllSpatialFiltering(start, end);
	//pic.AllTemporalFiltering();
	pic.AllTemporalFiltering3channel();

	printf("Press any key...");
	getchar();
	return 0;
}