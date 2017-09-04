#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){
    Mat img, gray;
    if( argc != 2 && !(img=imread(argv[1],1)).data)
        return -1;

    cvtColor(img, gray, CV_BGR2GRAY);
    GaussianBlur(gray, gray, Size(9, 9), 2, 2);
    vector<Vec3f> circles;
    HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 2, gray.rows/4, 200, 100);

    for( size_t i = 0; i < circles.size(); i++){
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle(img, center, 3, Scalar(0,255,0), -1, 8, 0);
        circle(img, center, radius, Scalar(0,0,255), 3, 8, 0);
    }

    namedWindow("circles", 1);
    imshow("circles", img);

	return 0;
}
