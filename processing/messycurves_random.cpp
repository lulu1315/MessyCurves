#include <iostream>
#include <fstream>
#include <cstdlib>
#include <time.h>

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "PerlinNoise.h"

using namespace std;
using namespace cv;

float PerlinOctave (int px, int py, int pz, float noiseScale, int octaves, PerlinNoise pn) {
    float amplitude = 1;
    float finalnoise = 0;
    float maxValue=0;
    float frequency = noiseScale;
    float persistence=.75;
    for (int o = 0; o < octaves; o++) {
        finalnoise+=pn.noise((float)px/frequency,(float)py/frequency,(float)pz)*amplitude;
        maxValue += amplitude;
        amplitude *=persistence;
        frequency *=.5;
        }
    return finalnoise/maxValue;
}
   
Point2f PixelForce(Point2f pos, Mat ima, float perception, float pixeldiv) {
    //pixels force
    int pcount = 0;
    Point2f target;
    target.x=0.0;
    target.y=0.0;
    int width =ima.cols;
    int height=ima.rows;
    for (int i = -floor(perception/2) ; i < perception/2 ; i++ ) {
        for (int j = -floor(perception/2) ; j < perception/2 ; j++ ) {
            if (i == 0 && j == 0) {
                continue;
            }
        int x = floor(pos.x+i);
        int y = floor(pos.y+j);
        if (x <= width-1 && x >= 0 && y < height-1 && y >= 0) {
            float b = (float)ima.at<uchar>(y, x);
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
    }
    if (pcount != 0) {
        return target/(float)pcount;
        }
    else {
        return target;
        }
}
    
Point2f NoiseForce(Point2f pos, float z, float noiseScale,float noiseInfluence, int octaves, PerlinNoise pn) {
    //add noise force
    float noise=PerlinOctave(pos.x,pos.y,z,noiseScale,octaves,pn);
    float remapnoise=noise*5*6.2831855;
    Point2f noisep;
    noisep.x=cos(remapnoise);
    noisep.y=sin(remapnoise);
    return noisep*noiseInfluence;
}
    
Point2f BoundForce(Point2f pos, float bound, float boundForceFactor, Mat ima) {
    //add bound force
    int width =ima.cols;
    int height=ima.rows;
    Point2f boundForce;
    boundForce.x=0.;
    boundForce.y=0.;
    if (pos.x < bound) {
        boundForce.x = (bound-pos.x)/bound;
        } 
    if (pos.x > width - bound) {
        boundForce.x = (pos.x - width)/bound;
        } 
    if (pos.y < bound) {
        boundForce.y = (bound-pos.y)/bound;
        } 
    if (pos.y > height - bound) {
        boundForce.y = (pos.y - height)/bound;
        } 
    return boundForce*boundForceFactor;
    }
    
Point2f RandomResetPos(int lumtreshold, Mat ima) {
    Point2f pos;
    pos.x=0.;
    pos.y=0.;
    bool hasFound = false;
    int width =ima.cols;
    int height=ima.rows;
    int b;
    while (!hasFound) {
        pos.x = (((float)rand())/(float)RAND_MAX)*width;
        pos.y = (((float)rand())/(float)RAND_MAX)*height;
        b=(int)ima.at<uchar>(floor(pos.y), floor(pos.x));
        if(b < lumtreshold) {
            hasFound = true;
            }
        }
    return pos;
}

float mag(Point2f vec) { //magnitude of vector
    return sqrt((vec.x*vec.x)+(vec.y*vec.y));
}

int main(int argc, const char* argv[])
{
    //random seed
    srand (33.333);
    //perlin noise
    PerlinNoise pn;
    
    string image_in_name;
    string image_out_name;
    image_in_name   = argv[1];
    image_out_name  = argv[2];
    
    Mat ima_in = imread(image_in_name, IMREAD_GRAYSCALE);
    Mat overlay;
    cout << "image in : " << image_in_name << endl;
    
    Mat ima_messy(ima_in.rows, ima_in.cols, CV_8UC1, Scalar(255));
    
    //parameters
    int subStep=800;
    float z=0;
    float zstep=.02;
    int ccount=0;
    int count=0;
    int maxCount=100;
    float maxSpeed=3.0;
    float perception=10;
    float pixeldiv=100.;
    float bound=60;
    float boundForceFactor=0.16;
    float noiseScale=100.0;
    float noiseInfluence=1/20.0;
    int noiseoctaves=1;
    float lineopacity=.2;
    int lumtreshold=30;
    //
    int width =ima_in.cols;
    int height=ima_in.rows;
    cout << "width/height : " << width << "/" << height << endl;
    //initialize
    Point2f p(((float)width)/2,((float)height)/2);
    Point2f ppos=p;
    Point2f pos=p;
    Point2f vel(0.0,0.0);
    Point2f force(0.0,0.0);
    int shade=0;
    bool isStop = false;
    
while (!isStop) {
    pos=RandomResetPos(lumtreshold,ima_in);
    //cout << "reset : " << pos.x << "," << pos.y << endl;
    for(int step=0; step<subStep; step++)
        {
        //update
        ppos=pos;
        force=force*0.0;
        
        //add pixels force
        force=force+PixelForce(pos,ima_in,perception,pixeldiv);
        
        //add noise force
        float forcenorm=mag(force);
        if( forcenorm < 0.01) {
            force=force+(NoiseForce(pos,z,noiseScale,noiseInfluence,noiseoctaves,pn)*5);
            }  
        else {
            force=force+NoiseForce(pos,z,noiseScale,noiseInfluence,noiseoctaves,pn);
            }
        
        //add bound force
        force=force+BoundForce(pos,bound,boundForceFactor,ima_in);
        
        vel=vel+force;
        vel=vel*0.9999; //wtf?
        float normvel=mag(vel);
        if (normvel > maxSpeed) {
            vel=vel*(maxSpeed/normvel);
            }
    
        pos=pos+vel;
        
        if (pos.x > width || pos.x < 0 || pos.y > height || pos.y < 0) {
            count = 0;
            //reseting pos;
            pos=RandomResetPos(lumtreshold,ima_in);
            //cout << "reset : " << pos.x << "," << pos.y << endl;
            ppos=pos;
            vel=vel*0;
        }
        
        //this.show();
        count++;
        if (count > maxCount) {
            count = 0;
            //this.reset();
            pos=RandomResetPos(lumtreshold,ima_in);
            //cout << "reset : " << pos.x << "," << pos.y << endl;
            ppos=pos;
            vel=vel*0;
            }
        
        //overlay slows down the whole shit .... damn!
        ima_messy.copyTo(overlay);
        line(overlay,ppos,pos,(shade),1,CV_AA);
        addWeighted(overlay, lineopacity,ima_messy,1-lineopacity,0,ima_messy);
        z+=zstep;
        }
        
    ccount++;
    cout << "ccount : " << ccount << endl;
	if (ccount > width) {
		isStop = true;
	}
}
    cout << "writing : " << image_out_name << endl;
    imwrite(image_out_name,ima_messy);
    return 0;
}

/*
 * https://answers.opencv.org/question/68215/how-do-i-cast-point2f-to-point2d/
*/

    /*
    //visualize perlin noise
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            ima_messy.at<uchar>(i, j)=255.*PerlinOctave(i,j,0,noiseScale,8,pn);
        }
    }
    imwrite("perlin.png",ima_messy);
    */
    
    /*
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            ima_messy.at<uchar>(i, j)=ima_in.at<uchar>(i, j);
        }
    }
    imwrite("input.png",ima_messy);
    */
