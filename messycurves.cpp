#include <iostream>
#include <fstream>

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

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
    
    Mat ima_messy = ima_in.clone();
    //parameters
    int subStep =800;
    float z     =0;
    int count   =0;
    //
    int width=ima_messy.cols;
    int height=ima_messy.rows;
    cout << "width/height : " << width << "/" << height << endl;
    
    cout << "writing : " << image_out_name << endl;
    imwrite(image_out_name,ima_messy);
    return 0;
}
