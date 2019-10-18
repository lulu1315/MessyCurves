#include <cstdlib>

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Forces.cpp"

using namespace std;
using namespace cv;

Mat ShowPixelForce(Mat ima, int sampling, float vscale, float halfperception, float pixeldiv ,float &maxforce) {
    int width =ima.cols;
    int height=ima.rows;
    Mat matforce(ima.rows, ima.cols, CV_8UC1, Scalar(255));
    Point2f getforce,pos;
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            pos.x=j;
            pos.y=i;
            getforce=PixelForce(pos,ima,halfperception,pixeldiv);
            if (mag(getforce) > maxforce) {maxforce = mag(getforce);}
            if (!(i%sampling) && !(j%sampling)) {
                line(matforce,pos,pos+(getforce*vscale),(0),1,CV_AA);
                }
        }
    }
    return matforce;
}

Mat ShowPixelForceAngle(Mat ima, float halfperception, float pixeldiv ,float &maxforce) {
    int width =ima.cols;
    int height=ima.rows;
    Mat dx(ima.rows, ima.cols, CV_32F, 1.);
    Mat dy(ima.rows, ima.cols, CV_32F, 1.);
    Mat force_angle(ima.rows, ima.cols, CV_32F, 1.);
    Mat force_mag(ima.rows, ima.cols, CV_32F, 1.);
    
    Point2f getforce,pos;
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            pos.x=j;
            pos.y=i;
            getforce=PixelForce(pos,ima,halfperception,pixeldiv);
            dx.at<float>(i, j)=getforce.x;
            dy.at<float>(i, j)=getforce.y;
            //cout << "pixforce : " << getforce << endl;
        }
    }
    cartToPolar(dx, dy, force_mag, force_angle, 1);
    force_angle=force_angle*180./360.;
    //convert to color
    Mat h(ima.rows, ima.cols, CV_8UC1, Scalar(0));
    Mat s(ima.rows, ima.cols, CV_8UC1, Scalar(255));
    Mat v(ima.rows, ima.cols, CV_8UC1, Scalar(255));
    Mat hsv(ima.rows, ima.cols, CV_8UC3);
    Mat bgr(ima.rows, ima.cols, CV_8UC3);
    force_angle.convertTo(h,CV_8UC1);
    vector<Mat> channels;
    channels.push_back(h);
    channels.push_back(s);
    channels.push_back(v);
    merge(channels, hsv);
    cvtColor(hsv, bgr, CV_HSV2BGR);
    return bgr;
}

Mat ShowCurlForce(Mat ima, int sampling, float vscale, float &maxforce) {
    int width =ima.cols;
    int height=ima.rows;
    Mat matforce(ima.rows, ima.cols, CV_8UC1, Scalar(255));
    Point2f getforce,pos;
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            pos.x=(float)j;
            pos.y=(float)i;
            getforce=CurlForce(ima,pos);
            if (mag(getforce) > maxforce) {maxforce = mag(getforce);}
            if (!(i%sampling) && !(j%sampling)) {
                line(matforce,pos,pos+(getforce*vscale),(0),1,CV_AA);
                }
        }
    }
    return matforce;
}

Mat ShowCurlNoiseForce(Mat ima, int sampling, float vscale, float z, float k, FastNoise fn,float &maxforce) {
    int width =ima.cols;
    int height=ima.rows;
    Mat matforce(ima.rows, ima.cols, CV_8UC1, Scalar(255));
    Point2f getforce,pos;
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            pos.x=j;
            pos.y=i;
            getforce=CurlNoiseForce(pos,z,k,fn);
            if (mag(getforce) > maxforce) {maxforce = mag(getforce);}
            if (!(i%sampling) && !(j%sampling)) {
                line(matforce,pos,pos+(getforce*vscale),(0),1,CV_AA);
                }
        }
    }
    return matforce;
}

Mat ShowFlowForce(Mat flow, int sampling, float vscale, float &maxforce) {
    int width =flow.cols;
    int height=flow.rows;
    Mat matforce(height, width, CV_8UC1, Scalar(255));
    Point2f getforce,pos;
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            pos.x=j;
            pos.y=i;
            getforce=FlowForce(pos,flow);
            if (mag(getforce) > maxforce) {maxforce = mag(getforce);}
            if (!(i%sampling) && !(j%sampling)) {
                line(matforce,pos,pos+(getforce*vscale),(0),1,CV_AA);
                }
        }
    }
    return matforce;
}

Mat ShowFastNoiseForce(Mat ima, int sampling, float vscale, float z, float k, float circularboost, FastNoise fn , bool curl, float &maxforce) {
    int width =ima.cols;
    int height=ima.rows;
    Mat matforce(ima.rows, ima.cols, CV_8UC1, Scalar(255));
    Point2f getforce,pos;
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            pos.x=j;
            pos.y=i;
            if (!curl) {getforce=FastNoiseForce(pos,z,k,circularboost,fn);}
            else {getforce=CurlNoiseForce(pos,z,k,fn);}
            if (mag(getforce) > maxforce) {maxforce = mag(getforce);}
            if (!(i%sampling) && !(j%sampling)) {
                line(matforce,pos,pos+(getforce*vscale),(0),1,CV_AA);
                }
        }
    }
    return matforce;
}

Mat ShowSuperEllipseForce(Mat ima, int sampling, float vscale, float bound, float n, float &maxforce) {
    int width =ima.cols;
    int height=ima.rows;
    Mat matforce(ima.rows, ima.cols, CV_8UC1, Scalar(255));
    Point2f getforce,pos;
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            pos.x=j;
            pos.y=i;
            getforce=SuperEllipseForce(pos,bound,n,ima);
            if (mag(getforce) > maxforce) {maxforce = mag(getforce);}
            if (!(i%sampling) && !(j%sampling)) {
                line(matforce,pos,pos+(getforce*vscale),(0),1,CV_AA);
                }
        }
    }
    return matforce;
}

Mat ShowGradientForce(Mat dx,Mat dy, int sampling, float vscale, float &maxforce) {
    int width =dx.cols;
    int height=dx.rows;
    Mat matforce(dx.rows, dx.cols, CV_8UC1, Scalar(255));
    Point2f getforce,pos;
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            pos.x=j;
            pos.y=i;
            getforce=GradientForce(dx,dy,pos);
            if (mag(getforce) > maxforce) {maxforce = mag(getforce);}
            if (!(i%sampling) && !(j%sampling)) {
                line(matforce,pos,pos+(getforce*vscale),(0),1,CV_AA);
                }
        }
    }
    return matforce;
}
