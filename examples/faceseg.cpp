#include <opencv2/opencv.hpp>
#include "net.h"
#include <string>
#include <vector>
#include <time.h>
#include <algorithm>
#include <map>
#include <iostream>

using namespace std;
using namespace cv;

const int part_colors[][3] = { { 0, 0,255 },{ 255, 85, 0 },{ 255, 170, 0 },
{ 255, 0, 85 },{ 255, 0, 17 },
{ 0, 255, 0 },{ 0, 255, 255 },{ 255, 0, 255 }};
int main(int argc, char** argv)
{
	ncnn::Net net;

	net.load_param("segmodel.param");
	net.load_model("segmodel.bin");
	
	cv::Mat src = cv::imread("20200513153020.jpg");

	ncnn::Extractor ex = net.create_extractor();
	ncnn::Mat ncnn_in = ncnn::Mat::from_pixels_resize(src.data,
		ncnn::Mat::PIXEL_BGR2RGB, src.cols, src.rows,256,256);
	const float meanVals[3] = { 123.675f, 116.28f,  103.53f };
	const float normVals[3] = { 0.01712475f, 0.0175f, 0.01742919f };
	ncnn_in.substract_mean_normalize(meanVals, normVals);

	ex.set_light_mode(true);
	ex.set_num_threads(2);

	ex.input("input", ncnn_in);

	ncnn::Mat mask;
	ex.extract("output", mask);
	float *mask_data = (float*)mask.data;
	cv::Mat segidx = cv::Mat::zeros(256, 256, CV_8UC1);
		
	unsigned char *segidx_data = segidx.data;
	int h = mask.h;
	int w = mask.w;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			int maxk = 0;
			float tmp = mask_data[0 * w * h + i * w + j];
			for (int k = 0; k < 8; k++)
			{
				if (tmp < mask_data[k * w * h + i * w + j])
				{
					tmp = mask_data[k * w * h + i * w + j];
					maxk = k;
				}
			}
			segidx_data[i * w + j] = maxk;
		}
	}
	cv::Mat result = cv::Mat::zeros(256, 256, CV_8UC3);
	for (int i = 0; i < segidx.rows; i++)
	{
		for (int j = 0; j < segidx.cols; j++)
		{
			int indx = segidx.at<uchar>(i, j);
			if (indx == 0)
				continue;
			result.at<cv::Vec3b>(i, j) = cv::Vec3b(part_colors[indx][0], part_colors[indx][1], part_colors[indx][2]);
		}
	}
		
	cv::imshow("result", result);
	cv::waitKey();
	

	return 0;
}
