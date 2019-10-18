#include <cstdlib>

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "tinysplinecpp.h"

using namespace std;
using namespace cv;

void drawCurve(Mat &overlay, vector<Point2f> pointList, float ratio, float splinestep, float size, float colorpower) {
    bool forcelines=false;
    float decreasesize,increasecolor;
    if ((pointList.size() > 3) && (!forcelines)) {  //use bsplines
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
            decreasesize=size*(1-u);
            increasecolor=pow(u,colorpower);
            line(overlay,splinepointi,splinepointj,((int)255.*increasecolor),size,CV_AA);
            //line(overlay,splinepointi,splinepointj,(0),size,CV_AA);
            //circle(overlay,splinepoint,1,(0),1,CV_AA);
            }
        }
    else
        {
        for (int np = 0; np < pointList.size()-1; np++)
            {
            line(overlay,pointList[np]*ratio,pointList[np+1]*ratio,(0),size,CV_AA);
            //circle(overlay,pointList[np]*ratio,2,(20),1,CV_AA);
            }
        }
    return;
}
