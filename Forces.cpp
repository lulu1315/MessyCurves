#include <cstdlib>

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "FastNoise.h"
#include "Maths.cpp"

using namespace std;
using namespace cv;

double colineardamping(Point2f vec1, Point2f vec2) {
    //normalize vectors
    float mydot;
    float mypower=1;
    if (mag(vec1) == 0 || mag(vec2) == 0)  {
        mydot=1.;
    }
    else {
        Point2f vec1norm=vec1/mag(vec1);
        Point2f vec2norm=vec2/mag(vec2);
        mydot=vec1norm.dot(vec2norm); //between 1 and -1
        //mydot=range(mydot,-1,1,1,0); //reversed between 0 and 1
        //cout << "dot : " << mydot << endl;
        //cout << "dot : " << mydot << " " << vec1norm << " " << vec2norm << endl;
    }
    return mydot;
}

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

Point2f FlowForce(Point2f pos, Mat flow) {
    Point2f flowvec = flow.at<Point2f>(pos.y,pos.x);
    return flowvec;
}

Point2f FastNoiseForce(Point2f pos, float z, float k, float circularboost, FastNoise fn) {
    //add noise force
    float noise=(sigmoid(fn.GetNoise(pos.x,pos.y,z),k));
    //float noise=(power(fn.GetNoise(pos.x,pos.y,z),k)+1.)/2.;
    float remapnoise=noise*circularboost*6.2831855;
    Point2f noisep;
    noisep.x=cos(remapnoise);
    noisep.y=sin(remapnoise);
    //dividing by 5 to be somehow coherent with the other forces
    return noisep/10.;
}

//calculate curl noise
Point2f CurlNoiseForce(Point2f pos, float z, float k, FastNoise fn) {
    Point2f curlvec;
    //float eps = 0.0001;
    float eps = 1;
    float n1,n2;
    
    //Find rate of change in X direction
    n1 = sigmoid(fn.GetNoise(pos.x+eps,pos.y,z),k);
    n2 = sigmoid(fn.GetNoise(pos.x-eps,pos.y,z),k);
    //Average to find approximate derivative
    float b = (n1 - n2)/(2. * eps);

    //Find rate of change in Y direction
    n1 = sigmoid(fn.GetNoise(pos.x,pos.y+eps,z),k);
    n2 = sigmoid(fn.GetNoise(pos.x,pos.y-eps,z),k);

    //Average to find approximate derivative
    float a = (n1 - n2)/(2. * eps);
    //cout << "a,b : " << a << "," << b << endl;
    //Curl
    curlvec.x=a;
    curlvec.y=-b;
    return curlvec*10;
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

//calculate curl vector of a scalarfield
Point2f CurlForce(Mat scalarfield, Point2f vec) {
    Point2f curlvec;
    int eps = 1;
    float n1,n2;
    
    //Find rate of change in X direction
    n1 = (float)scalarfield.at<uchar>(vec.y, vec.x + eps);
    n2 = (float)scalarfield.at<uchar>(vec.y, vec.x - eps);
    n1 = ((n1/255.)*2.)-1.;
    n2 = ((n2/255.)*2.)-1.;
    //Average to find approximate derivative
    float b = (n1 - n2)/(2. * (float)eps);

    //Find rate of change in Y direction
    n1 = (float)scalarfield.at<uchar>(vec.y + eps, vec.x);
    n2 = (float)scalarfield.at<uchar>(vec.y - eps, vec.x);
    n1 = ((n1/255.)*2.)-1.;
    n2 = ((n2/255.)*2.)-1.;

    //Average to find approximate derivative
    float a = (n1 - n2)/(2. * (float)eps);
    //cout << "a,b : " << a << "," << b << endl;
    //Curl
    curlvec.x=a;
    curlvec.y=-b;
    return curlvec;
}

//calculate gradient force from sobels dx,dy
Point2f GradientForce(Mat dx, Mat dy, Point2f pos) {
    Point2f force;
    force.x=dx.at<float>(pos.y, pos.x)/(sqrt(20)*255.);
    force.y=dy.at<float>(pos.y, pos.x)/(sqrt(20)*255.);
    return force;
}
