#include <cstdlib>

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Forces.cpp"

using namespace std;
using namespace cv;

Mat ShowPixelForce(Mat ima, int sampling, float vscale, float halfperception, float pixeldiv) {
    int width =ima.cols;
    int height=ima.rows;
    Mat pixelforce(ima.rows, ima.cols, CV_8UC1, Scalar(255));
    Point2f target,pos;
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            pos.x=j;
            pos.y=i;
            target=PixelForce(pos,ima,halfperception,pixeldiv)*vscale;
            if (!(i%sampling) && !(j%sampling)) {
                line(pixelforce,pos,pos+target,(0),1,CV_AA);
                }
        }
    }
    return pixelforce;
}

Mat ShowFastNoiseForce(Mat ima, int sampling, float vscale, float z, float noiseInfluence, float k, float circularboost, FastNoise fn) {
    int width =ima.cols;
    int height=ima.rows;
    Mat noiseforce(ima.rows, ima.cols, CV_8UC1, Scalar(255));
    Point2f target,pos;
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            pos.x=j;
            pos.y=i;
            target=FastNoiseForce(pos,z,noiseInfluence,k,circularboost,fn)*vscale;
            if (!(i%sampling) && !(j%sampling)) {
                line(noiseforce,pos,pos+target,(0),1,CV_AA);
                }
        }
    }
    return noiseforce;
}

Mat ShowBoundForce(Mat ima, Mat mask, int sampling, float vscale, float bound, float boundForceFactor) {
    int width =ima.cols;
    int height=ima.rows;
    Mat boundforce(ima.rows, ima.cols, CV_8UC1, Scalar(255));
    Point2f target,pos;
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            pos.x=j;
            pos.y=i;
            target=BoundForce(pos,bound,boundForceFactor,ima,mask)*vscale;
            if (!(i%sampling) && !(j%sampling)) {
                line(boundforce,pos,pos+target,(0),1,CV_AA);
                }
        }
    }
    return boundforce;
}

Mat ShowSuperEllipseForce(Mat ima, int sampling, float vscale, float bound, float n) {
    int width =ima.cols;
    int height=ima.rows;
    Mat superellipseforce(ima.rows, ima.cols, CV_8UC1, Scalar(255));
    Point2f target,pos;
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            pos.x=j;
            pos.y=i;
            if (!(i%sampling) && !(j%sampling)) {
                target=SuperEllipseForce(pos,bound,n,ima)*vscale;
                //cout << "target : " << target.x << "," << target.y << endl;
                line(superellipseforce,pos,pos+target,(0),1,CV_AA);
                //circle(superellipseforce,pos,1,(20),1,CV_AA);
                }
        }
    }
    return superellipseforce;
}
