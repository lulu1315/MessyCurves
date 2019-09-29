#include <cstdlib>

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

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
