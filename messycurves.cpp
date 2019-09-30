#include <iostream>
#include <fstream>
#include <cstdlib>
#include <time.h>

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

//#include "tinysplinecpp.h"
#include "Mask.cpp"
#include "Show.cpp"
#include "Draw.cpp"


using namespace std;
using namespace cv;

int main(int argc, const char* argv[])
{
    FastNoise fn;
    
    string image_in_name;
    string image_out_name;
    image_in_name   = argv[1];
    image_out_name  = argv[2];
    
    Mat ima_in = imread(image_in_name, IMREAD_GRAYSCALE);
    cout << "reading : " << image_in_name << endl;
    
    //parameters
    int     strokes             =atoi(argv[3]); //2000
    int     maxCount            =atoi(argv[4]); //points in stroke
    float   maxSpeed            =20.;
    float   veldamping          =atof(argv[9]);
    //pixelforce
    float   pixelInfluence      =atof(argv[5]);
    float   tangentInfluence    =atof(argv[6]);
    int     halfperception      =2;
    float   pixeldiv            =100.;
    //fastnoiseforce
    bool    donoiseforce=true;
    float   noiseInfluence      =atof(argv[7]); //0.5
    float   zstep               =0.005;
    float   noisepower          =3; // sigmoid
    float   circularboost       =5;
    int     fnnoisetype         =3;
    int     fnseed              =1337;
    float   fnfrequency         =.002;
    int     fnoctaves           =4;
    float   fnlacunarity        =2;
    float   fngain              =0.5;
    //boundforce
    bool    doboundforce=true;
    float   bound=              150;
    float   boundForceFactor    =atof(argv[8]); //0.5
    float   boundSuperness      =5;
    //display
    float   lineopacity         =atof(argv[10]);
    int     lumtreshold         =60;
    float   splinestep          =0.01;
    //int     masksampling=50; //undersampling of mask
    int     output_x            =atoi(argv[11]);
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
    cout << "pixels under treshold (<" << lumtreshold << ") : " << maskcount << endl;
    if (strokes > maskcount) {strokes = maskcount;}
    int masksampling=round((float)maskcount/(float)strokes);
    cout << "pixels to draw : " << strokes << " (sampling " << masksampling << ")" << endl;
    strokes=round((float)maskcount/(float)masksampling);
    cout << "adjusted pixels to draw : " << strokes << endl;
    //initialize
    float   absSpeed=0.;
    int     clampedstroke=0;
    int     allstroke=0;
    float   z=0;
    //image size
    int width =ima_in.cols;
    int height=ima_in.rows;
    cout << "input  width/height : " << width << "/" << height  << endl;
    float   ratio=1.;
    if (output_x != 0) {
        ratio=(float)output_x/(float)width;
        cout << "resize ratio : " << ratio << endl;
        cout << "output width/height : " << width*ratio << "/" << height*ratio << endl;
        }
    
    Mat overlay;
    Mat ima_messy(floor(ima_in.rows*ratio), floor(ima_in.cols*ratio), CV_8UC1, Scalar(255));
    //
    Point2f ppos,pos;
    Point2f vel(0.0,0.0);
    Point2f force(0.0,0.0);
    Point2f pixelforce,tangentforce,noiseforce,boundforce;
    float maxpixforce=0.;
    float maxnoiseforce=0.;
    int mcount=0;
    
    //debug
    bool debug=false;
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
    Mat PixelForceMat=ShowPixelForce(ima_in,5,30,halfperception,pixeldiv,maxpixforce);
    imwrite("pixelforce.png",PixelForceMat);
    cout << "(debug) writing : pixelforce.png" << endl;
    cout << "(debug) max pixelforce magnitude : " << maxpixforce << endl;
    Mat NoiseForceMat=ShowFastNoiseForce(ima_in,5,10,z,noisepower,circularboost,fn,maxnoiseforce);
    imwrite("noiseforce.png",NoiseForceMat);
    cout << "(debug) writing : noiseforce.png" << endl;
    cout << "(debug) max noiseforce magnitude : " << maxnoiseforce << endl;
    Mat SuperEllipseForceMat=ShowSuperEllipseForce(ima_in,10,5,bound,boundSuperness);
    imwrite("superellipse.png",SuperEllipseForceMat);
    cout << "(debug) writing : superellipse.png" << endl;
    }
    
while (true) { //main loop
    ima_messy.copyTo(overlay);
    vector<Point2f> pointList;
    pos=MaskedResetPos(masksampling,PixelMask);
    pointList.push_back(pos);
    vel=vel*0;
    mcount++;
    //cout << "pos : " << pos.x << "," << pos.y << endl;
    for(int step=0; step<maxCount; step++) //stroke loop
        {
        //update
        ppos=pos;
        force=force*0.0;
        vel=vel*veldamping;
        //add gradient force
        pixelforce=PixelForce(pos,ima_in,halfperception,pixeldiv);
        force=force+(pixelInfluence*pixelforce);
        
        //add tangentforce
        tangentforce=ccwtangent(pixelforce);
        force=force+(tangentInfluence*tangentforce);
        
        //add noise force
        if (donoiseforce) {
            noiseforce=FastNoiseForce(pos,z,noisepower,circularboost,fn);
            force=force+(noiseInfluence*noiseforce);
        /*
        float forcenorm=mag(force);
        if( forcenorm < 0.01) {
            noiseforce=FastNoiseForce(pos,z,noisepower,circularboost,fn);
            force=force+(noiseInfluence*noiseforce*5);
            }  
        else {
            noiseforce=FastNoiseForce(pos,z,noisepower,circularboost,fn);
            force=force+(noiseInfluence*noiseforce);
            }
        */
        }
        
        
        
        //add bound force
        if (doboundforce) {
            //boundforce=BoundForce(pos,bound,boundForceFactor,ima_in,PixelMask);
            boundforce=SuperEllipseForce(pos,bound,boundSuperness,PixelMask);
            force=force+(boundForceFactor*boundforce);
            }
            
        vel=vel+force;
        
        float normvel=mag(vel);
        allstroke++;
        if (normvel > absSpeed) { absSpeed = normvel;}
        if (normvel > maxSpeed) { 
            vel=vel*(maxSpeed/normvel);
            absSpeed = maxSpeed;
            clampedstroke++;
        }
    
        pos=pos+vel;
        
        if (pos.x > width || pos.x < 0 || pos.y > height || pos.y < 0) {
            //line(overlay,ppos*ratio,pos*ratio,(0),1,CV_AA);
            pointList.push_back(pos);
            break;
        }
        
        //line(overlay,ppos*ratio,pos*ratio,(0),1,CV_AA);
        pointList.push_back(pos);
        //circle(overlay,ppos,2,(20),1,CV_AA);
        z+=zstep;
        }
        
    if (!(mcount%100)) {
        cout << "scribbling : " << mcount << "\r" << std::flush;
        }
      
    drawCurve(overlay,pointList,ratio,splinestep);
    /*
    //cout << "nbpoints : " << pointList.size() << endl;
    if (pointList.size() > 3) {  //use bsplines
        tinyspline::BSpline spline(pointList.size());
        vector<tinyspline::real> ctrlp = spline.controlPoints();
        int ctrpcount=0;
        //fill bspline
        for (int np = 0; np < pointList.size(); np++)
            {
            //circle(overlay,pointList[np]*ratio,1,(0),1,CV_AA);
            ctrlp[ctrpcount]  = pointList[np].x*ratio;
            ctrpcount++;
            ctrlp[ctrpcount]  = pointList[np].y*ratio;
            ctrpcount++;
            }
        spline.setControlPoints(ctrlp);
        //draw spline
        for (float u=0. ; u< 1.-splinestep ; u=u+splinestep)
            {
            vector<tinyspline::real> resulti = spline.eval(u).result();
            vector<tinyspline::real> resultj = spline.eval(u+splinestep).result();
            Point2f splinepointi,splinepointj;
            splinepointi.x=resulti[0];
            splinepointi.y=resulti[1];
            splinepointj.x=resultj[0];
            splinepointj.y=resultj[1];
            line(overlay,splinepointi,splinepointj,(0),1,CV_AA);
            //circle(overlay,splinepoint,1,(0),1,CV_AA);
            }
        }
    else
        {
        for (int np = 0; np < pointList.size()-1; np++)
            {
            line(overlay,pointList[np]*ratio,pointList[np+1]*ratio,(0),1,CV_AA);
            //circle(overlay,pointList[np]*ratio,2,(20),1,CV_AA);
            }
        }
    */
    addWeighted(overlay, lineopacity,ima_messy,1-lineopacity,0,ima_messy);
	if (mcount >= strokes) {
        //cout << "clampedstrokes : " << clampedstroke << endl;
        cout << "total strokes : " << allstroke << endl;
        int clampedstrokepercent=round(100.*(float)clampedstroke/(float)allstroke);
        cout << "maximum speed : " << absSpeed << " (" << clampedstrokepercent << "% (" << clampedstroke << ") strokes clamped over " << maxSpeed << ")" << endl;
        cout << "writing result : " << image_out_name << endl << endl;
        imwrite(image_out_name,ima_messy);
        return 0;
	}//end stroke
}//end while()
}//end main

//gmic blondie1.jpg -fx_normalize_local 2,6,5,40,1,11,0,50,50 -o blondie1_normalize.png
