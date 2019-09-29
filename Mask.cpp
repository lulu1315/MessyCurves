#include <cstdlib>

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

Point2f MaskedResetPos(int masksampling,Mat &Mask) {
    Point2f pos;
    pos.x=0.;
    pos.y=0.;
    int mcount=0;
    int width =Mask.cols;
    int height=Mask.rows;
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            if (Mask.at<uchar>(i, j) == 0)  {
                Mask.at<uchar>(i, j)=255;
                mcount ++;
                if (mcount == masksampling)  {
                pos.x=j;
                pos.y=i;
                return pos;
                break;
                }
            }
        }
    }
    //return pos;
}

Mat PreFillMask(Mat ima,int lumtreshold,int &maskcount) {
    int mcount=0;
    Mat PixelMask(ima.rows, ima.cols, CV_8UC1, Scalar(255));
    int width =ima.cols;
    int height=ima.rows;
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            if (ima.at<uchar>(i, j) < lumtreshold)  {
                PixelMask.at<uchar>(i, j)=0;
                mcount++;
                }
        }
    }
    maskcount=mcount;
    return PixelMask;
}
