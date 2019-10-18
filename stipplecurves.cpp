#include <iostream>
#include <fstream>
#include <cstdlib>
#include <time.h>

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/optflow.hpp>

#include "Mask.cpp"
#include "Show.cpp"
#include "Draw.cpp"
#include "PLY.cpp"


using namespace std;
using namespace cv;

int main(int argc, const char* argv[])
{
    FastNoise fn;
    
    string ply_in_name;
    string image_in_name;
    string image_out_name;
    //parameters
    ply_in_name                 = argv[1];
    image_in_name               = argv[2];
    image_out_name              = argv[3];
    int     maxCount            =atoi(argv[4]); //points in stroke
    //pixelforce
    bool    dopixelforce        =atoi(argv[5]);
    float   pixelInfluence      =atof(argv[6]);
    float   minpixforce         =atof(argv[7]);
    int     halfperception      =atoi(argv[8]); //2
    float   pixeldiv            =100.;
    //gradientforce,tangentforce from image fields
    bool    dogradientforce     =atoi(argv[9]);
    int     gradientblur        =atoi(argv[10]); //blur image for gradient
    float   gradientInfluence   =atof(argv[11]);
    bool    dotangentforce      =atoi(argv[12]);
    float   tangentInfluence    =atof(argv[13]);
    //dragforce
    float   dragInfluence       =atof(argv[14]);
    //fastnoiseforce
    bool    donoiseforce        =atoi(argv[15]);
    bool    docurlnoise         =atoi(argv[16]);
    float   noiseInfluence      =atof(argv[17]);
    float   zstep               =atof(argv[18]);
    float   noisepower          =3;
    float   circularboost       =5;
    int     fnnoisetype         =atoi(argv[19]); //4
    int     fnseed              =1337;
    float   fnfrequency         =atof(argv[20]); //0.005
    int     fnoctaves           =4;
    float   fnlacunarity        =2;
    float   fngain              =0.5;
    //boundforce
    bool    doboundforce        =atoi(argv[21]); //true
    float   bound               =atof(argv[22]); //80
    float   boundForceFactor    =atof(argv[23]);
    float   boundSuperness      =5;
    //colinear
    float   colinearlimit       =atof(argv[24]);
    //display
    float   maxSpeed            =atof(argv[25]); //2000
    float   lineopacity         =atof(argv[26]); //.5
    float   colorpower          =atof(argv[27]); //3
    float   sizemin             =atof(argv[28]); //1
    float   sizemax             =atof(argv[29]); //6
    float   linewidthmin        =atof(argv[30]); //0
    float   linewidthmax        =atof(argv[31]); //6
    float   splinestep          =atof(argv[32]); //.01
    float   oversampling        =atof(argv[33]); //2
    int     output_x            =atoi(argv[34]); //1920
    bool    debug               =atoi(argv[35]); //false
    
    //lets go
    Mat ima_in = imread(image_in_name, IMREAD_GRAYSCALE);
    cout << "reading : " << image_in_name << endl;
    int width =ima_in.cols;
    int height=ima_in.rows;
    cout << "input  width/height : " << width << "/" << height  << endl;
    
    //calculate gradient
    Mat dx,dy,ima_in_blur;
    GaussianBlur(ima_in,ima_in_blur,Size(gradientblur, gradientblur),0,0,BORDER_DEFAULT);
    Sobel(ima_in_blur, dx, CV_32F, 1,0);
    Sobel(ima_in_blur, dy, CV_32F, 0,1);
    Mat sobel_angle, sobel_mag;
    cartToPolar(dx, dy, sobel_mag, sobel_angle);
    
    //stipples
    vector<Point2f> stippleList;
    vector<float> stipplesizeList;
    cout << "reading : " << ply_in_name << endl;
    ReadPLY(ply_in_name,width,height,stippleList,stipplesizeList);
    
    int     strokes             =stippleList.size();
    cout << "pixels to draw : " << strokes << endl;
    if (1./maxCount < splinestep) {
        splinestep=1./maxCount;
        cout << "splinestep : " << splinestep << endl;
        }
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
    //initialize
    float   absSpeed=0.;
    int     clampedstroke=0;
    int     allstroke=0;
    float   z=0;
    float dragcoefficient;
    
    //image size
    cout << "input  width/height : " << width << "/" << height  << endl;
    float   ratio=1.;
    if (output_x != 0) {
        ratio=(float)output_x/(float)width;
        cout << "resize ratio : " << ratio << endl;
        cout << "output width/height : " << (int)(width*ratio) << "/" << (int)(height*ratio) << endl;
        }
    float   oversamplingratio=ratio*oversampling;
    cout << "oversampling ratio : " << oversamplingratio << endl;
    cout << "oversampling width/height : " << (int)(width*oversamplingratio) << "/" << (int)(height*oversamplingratio) << endl;
        
    Mat overlay;
    Mat ima_messy(floor(height*oversamplingratio), floor(width*oversamplingratio), CV_8UC1, Scalar(255));
    //
    Point2f ppos,pos;
    Point2f vel(0.0,0.0);
    Point2f force(0.0,0.0);
    Point2f previousforce(0.0,0.0);
    Point2f pixelforce,gradientforce,tangentforce,noiseforce,boundforce,dragforce;
    float stipplesize;
    int mcount=0;
    
    //see pixelforce animated
    /*
    float maxpixforce=0.;
    Mat PixelForceMat=ShowPixelForce(ima_in,5,30,halfperception,pixeldiv,maxpixforce);
    string imanumber = argv[];
    string finalname = "pixelforce." + imanumber + ".png";
    cout << "ima" << finalname << endl;
    imwrite(finalname,PixelForceMat);
    */
    
    //debug
    if (debug) {
    int sampling=5;
    int vscale=100;
    //visualizing fastnoise
    Mat fastnoisemat(height, width, CV_8UC1, Scalar(0));
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            fastnoisemat.at<uchar>(i, j)=(sigmoid(fn.GetNoise(j,i,1),noisepower))*255.;
        }
    }
    cout << "(debug) writing : fastnoise.png" << endl;
    imwrite("fastnoise.png",fastnoisemat);
    //visualizing pixel force
    float maxpixforce=0.;
    Mat PixelForceMat=ShowPixelForce(ima_in,sampling,vscale,halfperception,pixeldiv,maxpixforce);
    imwrite("pixelforce.png",PixelForceMat);
    cout << "(debug) writing : pixelforce.png" << endl;
    cout << "(debug) max pixelforce magnitude : " << maxpixforce << endl;
    Mat PixelForceAngleMat=ShowPixelForceAngle(ima_in,halfperception,pixeldiv,maxpixforce);
    imwrite("pixelforceangle.png",PixelForceAngleMat);
    cout << "(debug) writing : pixelforceangle.png" << endl;
    //visualizing noise force
    float maxnoiseforce=0.;
    Mat NoiseForceMat=ShowFastNoiseForce(ima_in,sampling,vscale,z,noisepower,circularboost,fn,docurlnoise,maxnoiseforce);
    if (docurlnoise) {
        imwrite("noiseforce.png",NoiseForceMat);
        cout << "(debug) writing : (curl) noiseforce.png" << endl;
        }
    else {
        imwrite("noiseforce.png",NoiseForceMat);
        cout << "(debug) writing : (nocurl) noiseforce.png" << endl;
        }
    cout << "(debug) max noiseforce magnitude : " << maxnoiseforce << endl;
    //visualizing bounding force
    float maxboundforce=0.;
    Mat SuperEllipseForceMat=ShowSuperEllipseForce(ima_in,sampling,vscale,bound,boundSuperness,maxboundforce);
    imwrite("superellipse.png",SuperEllipseForceMat);
    cout << "(debug) writing : superellipse.png" << endl;
    cout << "(debug) max boundforce magnitude : " << maxboundforce << endl;
    //visualizing gradient force
    float maxgradientforce=0.;
    Mat GradientForceMat=ShowGradientForce(dx,dy,sampling,vscale,maxgradientforce);
    imwrite("gradient.png",GradientForceMat);
    cout << "(debug) writing : gradient.png" << endl;
    cout << "(debug) max gradientforce magnitude : " << maxgradientforce << endl;
    //visualising gradient norm
    imwrite("gradientnorm.png",sobel_mag);
    cout << "(debug) writing : gradientnorm.png" << endl;
    //visualizing curl force
    float maxcurlforce=0.;
    Mat CurlForceMat=ShowCurlForce(ima_in_blur,sampling,vscale,maxcurlforce);
    imwrite("curlforce.png",CurlForceMat);
    cout << "(debug) writing : curlforce.png" << endl;
    cout << "(debug) max curlforce magnitude : " << maxcurlforce << endl;
    }

//forward stroke
for(int st=0; st<strokes; st++) {//main loop
    ima_messy.copyTo(overlay);
    vector<Point2f> pointList;
    pos=stippleList[st];
    //circle(overlay,pos*ratio,5,(0),1,CV_AA);
    stipplesize=range(stipplesizeList[st],sizemin,sizemax,linewidthmin,linewidthmax);
    //stipplesize=1;
    pointList.push_back(pos);
    vel=vel*0;
    mcount++;
    
    //forward strokes
    for(int step=0; step<maxCount; step++) //stroke loop
        {
        //update
        ppos=pos;
        force=force*0.0;
        
        //dampen vel
        dragcoefficient=pow((1.-((float)step/(float)(maxCount-1))),dragInfluence);
        vel=vel*dragcoefficient;
    
        //add pixel force
        if (dopixelforce) {
            pixelforce=PixelForce(pos,ima_in_blur,halfperception,pixeldiv);
            if (mag(pixelforce) < minpixforce) {pixelforce=pixelforce*0.;}
            force=force+(pixelInfluence*pixelforce);
        }
        
        if (dogradientforce) {
            //gradientforce=FlowForce(pos,gradientflow);
            gradientforce=GradientForce(dx,dy,pos);
            force=force+(gradientInfluence*gradientforce);
        }
        
        //add tangentforce
        if (dotangentforce) {
            //tangentforce=ccwtangent(gradientforce);
            tangentforce=CurlForce(ima_in_blur,pos);
            force=force+(tangentInfluence*tangentforce);
        }
        
        //add noise force
        if (donoiseforce) {
            if (docurlnoise) {
                noiseforce=CurlNoiseForce(pos,z,noisepower,fn);
                }
            else {
                noiseforce=FastNoiseForce(pos,z,noisepower,circularboost,fn);
                }
            force=force+(noiseInfluence*noiseforce);
        }
        
        //add bound force
        if (doboundforce) {
            boundforce=SuperEllipseForce(pos,bound,boundSuperness,ima_in);
            force=force+(boundForceFactor*boundforce);
            }
            
        vel=vel+force;
        
        float colinear=colineardamping(previousforce,vel);
        if (colinear > colinearlimit && step != 0) { 
            pointList.push_back(pos);
            break;
            //vel=vel*colineardamp;
        }
        previousforce=vel;
        
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
            pointList.push_back(pos);
            break;
        }
        
        pointList.push_back(pos);
        //circle(overlay,pos*ratio,2,(0),1,CV_AA);
        z+=zstep;
        }//end stroke
        
    if (!(mcount%100)) {
        cout << "forward scribbling : " << mcount << "\r" << std::flush;
        }
      
    drawCurve(overlay,pointList,oversamplingratio,splinestep,stipplesize,colorpower);
    addWeighted(overlay, lineopacity,ima_messy,1-lineopacity,0,ima_messy);
    }//end while()
    
//finished !
cout << "total strokes : " << allstroke << endl;
int clampedstrokepercent=round(100.*(float)clampedstroke/(float)allstroke);
cout << "maximum speed : " << absSpeed << " (" << clampedstrokepercent << "% (" << clampedstroke << ") strokes clamped over " << maxSpeed << ")" << endl;
//resize ima_messy
resize(ima_messy, ima_messy, Size(width*ratio,height*ratio), 0, 0, INTER_AREA);
cout << "writing result : " << image_out_name << endl << endl;
imwrite(image_out_name,ima_messy);
return 0;
}//end main

//gmic blondie1.jpg -fx_normalize_local 2,6,5,40,1,11,0,50,50 -o blondie1_normalize.png
//https://www.khanacademy.org/computing/computer-programming/programming-natural-simulations/programming-forces/a/newtons-laws-of-motion
