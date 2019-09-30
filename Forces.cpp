#include <cstdlib>

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "FastNoise.h"
#include "Maths.cpp"

using namespace std;
using namespace cv;

Point2f PixelForce(Point2f pos, Mat ima, int halfperception, float pixeldiv) {
    //pixels force
    int pcount = 0;
    Point2f target,gradient;
    float b;
    target.x=0.0;
    target.y=0.0;
    int width =ima.cols;
    int height=ima.rows;
    for (int i = -halfperception ; i <= halfperception ; i++ ) {
        for (int j = -halfperception ; j <= halfperception ; j++ ) {
            if (i == 0 && j == 0) {
                continue;
            }
        int x = floor(pos.x+i);
        int y = floor(pos.y+j);
        if (x<0) {x=0;}
        if (y<0) {y=0;}
        if (x>width-1) {x=width-1;}
        if (y>height-1) {y=height-1;}
            b = (float)ima.at<uchar>(y, x);
            b = 1.0 - (b/pixeldiv);
            float normp=sqrt(((float)i*(float)i)+((float)j*(float)j));
            Point2f normalizedp;
            normalizedp.x=(float)i/normp;
            normalizedp.y=(float)j/normp;
            target=target+(normalizedp*b/normp);
            //target.add(p.normalize().copy().mult(b).div(p.mag()));
            pcount++;
        }
    }
    if (pcount != 0) {
        gradient=2*target/(float)pcount;
        }
    else {
        gradient=target;
        }
    return gradient;
}

Point2f FastNoiseForce(Point2f pos, float z, float k, float circularboost, FastNoise fn) {
    //add noise force
    //float circularboost=15.;
    float noise=(sigmoid(fn.GetNoise(pos.x,pos.y,z),k));
    //float noise=(power(fn.GetNoise(pos.x,pos.y,z),k)+1.)/2.;
    float remapnoise=noise*circularboost*6.2831855;
    Point2f noisep;
    noisep.x=cos(remapnoise);
    noisep.y=sin(remapnoise);
    return noisep;
}

Point2f BoundForce(Point2f pos, float bound, float boundForceFactor, Mat ima, Mat mask) {
    //add bound force
    int width =ima.cols;
    int height=ima.rows;
    Point2f boundForce;
    boundForce.x=0.;
    boundForce.y=0.;
    float maskweight;
    if (pos.x < bound) {
        boundForce.x = (bound-pos.x)/bound;
        } 
    if (pos.x > width - bound) {
        boundForce.x = ((width - bound)-pos.x)/bound;
        } 
    if (pos.y < bound) {
        boundForce.y = (bound-pos.y)/bound;
        } 
    if (pos.y > height - bound) {
       boundForce.y = ((height - bound)-pos.y)/bound;
        } 
    maskweight=(float)mask.at<uchar>(pos.y, pos.x)/255.;
    return boundForce*boundForceFactor*maskweight;
    }
    
Point2f SuperEllipseForce(Point2f pos, float bound, float n, Mat mask) {
    //add superellipse force
    int width =mask.cols;
    int height=mask.rows;
    float maskweight;
    Point2f offsetp,target;
    offsetp.x=(pos.x*2)-(width-1);
    offsetp.y=(pos.y*2)-(height-1);
    float se=-(1.-(pow(abs(offsetp.x/(width-1-bound)),n)+pow(abs(offsetp.y/(height-1-bound)),n)));
    if (se < 0) {se=0;}
    offsetp=offsetp/mag(offsetp);
    maskweight=(float)mask.at<uchar>(pos.y, pos.x)/255.;
    target=-offsetp*se*maskweight;
    return target;
    }
