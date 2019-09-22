#include <iostream>
#include <fstream>

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "PerlinNoise.h"

using namespace std;
using namespace cv;

int main(int argc, const char* argv[])
{
    string image_in_name;
    string image_out_name;
    image_in_name   = argv[1];
    image_out_name  = argv[2];
    
    Mat ima_in = imread(image_in_name, IMREAD_GRAYSCALE);
    cout << "image in : " << image_in_name << endl;
    
    //Mat ima_messy = ima_in.clone();
    Mat ima_messy(ima_in.rows, ima_in.cols, CV_8UC1, Scalar(255));
    //parameters
    int subStep =800;
    float z     =0;
    int count   =0;
    int maxCount = 100;
    float maxSpeed=3.0;
    float perception=5;
    float bound=60;
    float boundForceFactor = 0.16;
    float noiseScale = 100.0;
    float noiseInfluence = 1 / 20.0;
    float dropRate = 0.004;
    float dropRange = 40;
    float dropAlpha = 150;
    float drawAlpha = 50;
    float drawWeight = 1;
    //
    int width=ima_in.cols;
    int height=ima_in.rows;
    cout << "width/height : " << width << "/" << height << endl;
    //perlin noise
    PerlinNoise pn;
    /*
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
            ima_messy.at<uchar>(i, j)=255.*pn.noise((float)i/noiseScale,(float)j/noiseScale, 0.55);
        }
    }
    imwrite("perlin.png",ima_messy);
    double noise = pn.noise(0.45, 0.8, 0.55);
    cout << "Perlin : " << noise << endl;
    */
    
    Point2f p(((float)width)/2,((float)height)/2);
    Point2f ppos=p;
    Point2f pos=p;
    Point2f vel(0.0,0.0);
    Point2f force(0.0,0.0);
    
    for(int step=0; step<subStep; step++)
        {
        //update
        force=force*0.0;
        //add pixels force
        Point2f target(0.0,0.0);
        count = 0;
        for (int i = -floor(perception/2) ; i < perception/2 ; i++ ) {
            for (int j = -floor(perception/2) ; j < perception/2 ; j++ ) {
                if (i == 0 && j == 0) {
                    continue;
                }
            float x = floor(pos.x+i);
            float y = floor(pos.y+j);
            if (x <= width-1 && x >= 0 && y < height-1 && y >= 0) {
                float b = (float)ima_in.at<uchar>(i, j);
                b = 1 - b/100.0;
                //Point2f lp(i, j);
                Point2f normp(i/sqrt((i*i)+(j*j)),j/sqrt((i*i)+(j*j)));
                target=target+((normp*b)/sqrt((i*i)+(j*j)));
                //target.add(p.normalize().copy().mult(b).div(p.mag()));
                count++;
                }
            }
        }
        if (count != 0) {
            //force.add(target.div(count));
            force=force+(target/(float)count);
            }
        
        //add noise force
        float noise=5*6.2831855*pn.noise(pos.x/noiseScale,pos.y/noiseScale,z);
        Point2f noisep(cos(noise),sin(noise));
        if(norm(noisep) < 0.01) {
            force=force+(noisep*noiseInfluence * 5);
            //force.add(p.mult(noiseInfluence * 5));
            }  
        else {
            force=force+(noisep*noiseInfluence);
            //force.add(p.mult(noiseInfluence));
            }
            
        //add bound force
        
        cout << "force : " << force.x << "," << force.y << endl;
        z+=0.01;
        }
    cout << "writing : " << image_out_name << endl;
    imwrite(image_out_name,ima_messy);
    return 0;
}
