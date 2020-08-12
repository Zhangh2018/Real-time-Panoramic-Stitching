#include "stdafx.h"
#include <opencv2/opencv.hpp>

void main()
{
	stitchImage("../A.JPG,../B.JPG,../C.JPG,../D.JPG");
	stitchCamera("01");
}