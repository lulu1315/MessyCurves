#include <cstdlib>
#include <iostream>
#include <fstream>

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

void ReadPLY(string plyfilename, int width, int height, vector<Point2f> &pointList, vector<float> &sizeList) {
    std::ifstream inply;
    //vector<Point2f> pointList;
    //read ply
    std::string plyword;
    inply.open (plyfilename);
    //find number of vertices
    while (plyword != "vertex")
        {
        inply >> plyword;
        }
        int nvertices;
        inply >> nvertices;
        cout << "reading PLY : " << plyfilename << " (" << nvertices << " points)" << endl;
        //continue in ply
        while (plyword != "end_header")
            {
            inply >> plyword;
            }
        //fill stippleList with ply values
        float xpos,ypos,zpos,size;
        Point2f p;
        for (int i=1;i<=nvertices;i++)
            {
            inply >> xpos;
            inply >> ypos;
            inply >> zpos;
            inply >> size;
            p.x=width*xpos;
            p.y=height*zpos;
            //cout << "stipple pos : " << stipple.x << "," << stipple.y << endl;
            pointList.push_back(p);
            sizeList.push_back(size);
            }
        inply.close();
        //return pointList;
}
