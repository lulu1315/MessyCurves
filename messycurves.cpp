#include <iostream>
#include <fstream>
#include <cstdlib>
#include <time.h>

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "PerlinNoise.h"
#include "FastNoise.h"

using namespace std;
using namespace cv;

float mag(Point2f vec) { //magnitude of vector
    return sqrt((vec.x*vec.x)+(vec.y*vec.y));
}

float sigmoid(float x,float k)
{
    return 1.0 / (1.0 + (float)exp(-k * x));
}

float power(float x,float k)
{
    if (x < 0) {
        return -pow(abs(x),k);
    }
    else {
        return pow(x,k);
    }
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
        gradient=target/(float)pcount;
        }
    else {
        gradient=target;
        }
    return gradient;
}
    
Point2f cwtangent(Point2f vec) {
    Point2f tangent;
    tangent.x=vec.y;
    tangent.y=-vec.x;
    return tangent;
}

Point2f ccwtangent(Point2f vec) {
    Point2f tangent;
    tangent.x=-vec.y;
    tangent.y=vec.x;
    return tangent;
}

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

Point2f FastNoiseForce(Point2f pos, float z, float noiseInfluence, float k, FastNoise fn) {
    //add noise force
    float noise=(sigmoid(fn.GetNoise(pos.x,pos.y,z),k));
    //float noise=(power(fn.GetNoise(pos.x,pos.y,z),k)+1.)/2.;
    float remapnoise=noise*5.*6.2831855;
    Point2f noisep;
    noisep.x=cos(remapnoise);
    noisep.y=sin(remapnoise);
    return noisep*noiseInfluence;
}

Mat ShowFastNoiseForce(Mat ima, int sampling, float vscale, float z, float noiseInfluence, float k, FastNoise fn) {
    int width =ima.cols;
    int height=ima.rows;
    Mat noiseforce(ima.rows, ima.cols, CV_8UC1, Scalar(255));
    Point2f target,pos;
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            pos.x=j;
            pos.y=i;
            target=FastNoiseForce(pos,z,noiseInfluence,k,fn)*vscale;
            if (!(i%sampling) && !(j%sampling)) {
                line(noiseforce,pos,pos+target,(0),1,CV_AA);
                }
        }
    }
    return noiseforce;
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

Point2f SuperEllipseForce(Point2f pos, float bound, float n, Mat mask) {
    //add superellipse force
    int width =mask.cols;
    int height=mask.rows;
    float maskweight;
    Point2f offsetp,target;
    offsetp.x=(pos.x*2)-(width);
    offsetp.y=(pos.y*2)-(height);
    float se=-(1.-(pow(abs(offsetp.x/(width-bound)),n)+pow(abs(offsetp.y/(height-bound)),n)));
    if (se < 0) {se=0;}
    offsetp=offsetp/mag(offsetp);
    maskweight=(float)mask.at<uchar>(pos.y, pos.x)/255.;
    target=-offsetp*se*maskweight;
    return target;
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
    cout << "pixels under treshold (<" << lumtreshold << ") : " << mcount << endl;
    maskcount=mcount;
    return PixelMask;
}

int main(int argc, const char* argv[])
{
    FastNoise fn;
    
    string image_in_name;
    string image_out_name;
    image_in_name   = argv[1];
    image_out_name  = argv[2];
    
    Mat ima_in = imread(image_in_name, IMREAD_GRAYSCALE);
    Mat overlay;
    cout << "image in : " << image_in_name << endl;
    
    Mat ima_messy(ima_in.rows, ima_in.cols, CV_8UC1, Scalar(255));
    
    //parameters
    float   zstep=0.001;
    int     maxCount=50;
    float   maxSpeed=10.0;
    //pixelforce
    float   pixelInfluence=10.;
    float   tangentInfluence=pixelInfluence/5;
    int     halfperception=2;
    float   pixeldiv=100.;
    //fastnoiseforce
    bool    donoiseforce=true;
    float   noiseInfluence=.05;
    float   noisepower=5; // sigmoid
    int     fnnoisetype=5;
    int     fnseed=1337;
    float   fnfrequency=.005;
    int     fnoctaves=3;
    float   fnlacunarity=2;
    float   fngain=0.5;
    //boundforce
    bool    doboundforce=true;
    float   bound=200;
    float   boundForceFactor=.5; //0.16;
    float   boundSuperness=2;
    //display
    float   lineopacity=.2;
    int     lumtreshold=80;
    int     masksampling=20; //undersampling of mask
    //initialize fastnoise
    //https://github.com/Auburns/FastNoise
    cout << "initializing fast noise"  << endl;
    fn.SetSeed(fnseed);
    fn.SetFrequency(fnfrequency);
    fn.SetFractalOctaves(fnoctaves);
    fn.SetFractalLacunarity(fnlacunarity);
    fn.SetFractalGain(fngain);
    switch(fnnoisetype) {
        case 0 : fn.SetNoiseType(FastNoise::Value);break;
        case 1 : fn.SetNoiseType(FastNoise::ValueFractal);break;
        case 2 : fn.SetNoiseType(FastNoise::Perlin);break;
        case 3 : fn.SetNoiseType(FastNoise::PerlinFractal);break;
        case 4 : fn.SetNoiseType(FastNoise::Simplex);break;
        case 5 : fn.SetNoiseType(FastNoise::SimplexFractal);break;
        case 6 : fn.SetNoiseType(FastNoise::Cellular);break;
        case 7 : fn.SetNoiseType(FastNoise::WhiteNoise);break;
        case 8 : fn.SetNoiseType(FastNoise::Cubic);break;
        case 9 : fn.SetNoiseType(FastNoise::CubicFractal);break;
        }
    //prefill mask
    int maskcount=0;
    Mat PixelMask=PreFillMask(ima_in,lumtreshold,maskcount);
    maskcount=floor(maskcount/masksampling);
    cout << "pixels to draw : " << maskcount << " (sampling " << masksampling << ")" << endl;
    
    //initialize
    float   z=0;
    int width =ima_in.cols;
    int height=ima_in.rows;
    cout << "width/height : " << width << "/" << height << endl;
    Point2f ppos,pos;
    Point2f vel(0.0,0.0);
    Point2f force(0.0,0.0);
    Point2f pixelforce,tangentforce,boundforce;
    int mcount=0;
    
    //debug
    bool debug=true;
    
    if (debug) {
    //visualizing pixelmask
    imwrite("pixelmask.png",PixelMask);
    cout << "(debug) writing : pixelmask.png" << endl;
    //visualizing fastnoise
    Mat fastnoisemat(ima_in.rows, ima_in.cols, CV_8UC1, Scalar(0));
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            //fastnoisemat.at<uchar>(i, j)=((fn.GetNoise(j,i,1)+1)/2)*255.;
            //fastnoisemat.at<uchar>(i, j)=(power(fn.GetNoise(j,i,1),noisepower)+1)/2*255.;
            fastnoisemat.at<uchar>(i, j)=(sigmoid(fn.GetNoise(j,i,1),noisepower))*255.;
        }
    }
    imwrite("fastnoise.png",fastnoisemat);
    cout << "(debug) writing : fastnoise.png" << endl;
    //visualizing forces
    Mat PixelForceMat=ShowPixelForce(ima_in,5,40,halfperception,pixeldiv);
    imwrite("pixelforce.png",PixelForceMat);
    cout << "(debug) writing : pixelforce.png" << endl;
    Mat NoiseForceMat=ShowFastNoiseForce(ima_in,5,40,z,noiseInfluence,noisepower,fn);
    imwrite("noiseforce.png",NoiseForceMat);
    cout << "(debug) writing : noiseforce.png" << endl;
    Mat BoundForceMat=ShowBoundForce(ima_in,PixelMask,5,40,bound,boundForceFactor);
    imwrite("boundforce.png",BoundForceMat);
    cout << "(debug) writing : boundforce.png" << endl;
    Mat SuperEllipseForceMat=ShowSuperEllipseForce(ima_in,10,5,bound,boundSuperness);
    imwrite("superellipse.png",SuperEllipseForceMat);
    cout << "(debug) writing : superellipse.png" << endl;
    }
    
while (true) {
    ima_messy.copyTo(overlay);
    pos=MaskedResetPos(masksampling,PixelMask);
    vel=vel*0;
    mcount++;
    //cout << "pos : " << pos.x << "," << pos.y << endl;
    for(int step=0; step<maxCount; step++)
        {
        //update
        ppos=pos;
        force=force*0.0;
        
        //add gradient force
        pixelforce=PixelForce(pos,ima_in,halfperception,pixeldiv);
        force=force+(pixelInfluence*pixelforce);
        
        //add tangentforce
        tangentforce=ccwtangent(pixelforce);
        force=force+(tangentInfluence*tangentforce);
        
        //add noise force
        if (donoiseforce) {
        float forcenorm=mag(force);
        if( forcenorm < 0.01) {
            force=force+(FastNoiseForce(pos,z,noiseInfluence,noisepower,fn)*5);
            }  
        else {
            force=force+FastNoiseForce(pos,z,noiseInfluence,noisepower,fn);
            }
        }
        
        //add bound force
        if (doboundforce) {
            //boundforce=BoundForce(pos,bound,boundForceFactor,ima_in,PixelMask);
            boundforce=SuperEllipseForce(pos,bound,boundSuperness,PixelMask);
            force=force+(boundForceFactor*boundforce);
            }
            
        vel=vel+force;
        vel=vel*0.9999; //wtf?
        float normvel=mag(vel);
        if (normvel > maxSpeed) {
            vel=vel*(maxSpeed/normvel);
            //cout << "clamping speed from " << normvel << " to " << mag(vel) << endl;
            }
    
        pos=pos+vel;
        
        if (pos.x > width || pos.x < 0 || pos.y > height || pos.y < 0) {
            line(overlay,ppos,pos,(0),1,CV_AA);
            break;
        }
        
        line(overlay,ppos,pos,(0),1,CV_AA);
        //circle(overlay,ppos,2,(20),1,CV_AA);
        z+=zstep;
        }
        
    if (!(mcount%100)) {
        cout << "scribbling : " << mcount << "\r" << std::flush;
        }
    
    addWeighted(overlay, lineopacity,ima_messy,1-lineopacity,0,ima_messy);
	if (mcount >= maskcount) {
        cout << "writing result : " << image_out_name << endl;
        imwrite(image_out_name,ima_messy);
        return 0;
	}//end for
}//end while()
}//end main
