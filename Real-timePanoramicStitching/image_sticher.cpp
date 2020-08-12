#include "stdafx.h"
#include "windows.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


void stitchImage(string imgNames)
{
	Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);
	vector<Mat> imgs;
	Mat pano;
	Mat img;

	size_t lastPos = imgNames.find_first_not_of(',', 0);
	size_t pos = imgNames.find(',', lastPos);
	while (lastPos != string::npos)
	{
		string imgName = imgNames.substr(lastPos, pos - lastPos);
		lastPos = imgNames.find_first_not_of(',', pos);
		pos = imgNames.find(',', lastPos);
		img = imread(imgName);
		if (img.empty())
		{
			cout << "Can't read " << imgName << endl;
			continue;
		}
		imgs.push_back(img);
	}

	cout << "Start processing" << endl;
	long t0 = GetTickCount();
	Stitcher::Status status = stitcher->stitch(imgs, pano);
	long t1 = GetTickCount();
	cout << "Time Cost: " << t1 - t0 << "ms" << endl;

	if (status != Stitcher::OK)
	{
		cout << "Can't stitch images, error code = " << int(status) << endl;
	}
	else
	{
		imwrite("../Panorama.jpg", pano);
		cout << "Stitch finished" << endl;
	}
}