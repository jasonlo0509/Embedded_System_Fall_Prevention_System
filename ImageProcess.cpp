#include "ImageProcess.h"
#include "mainWindow.h"
#include "xbee.h"
#include "ui_subwindow.h"
#include <QLabel>
#include <QDebug>
#include <QTextStream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#define PI 3.1415926

using namespace cv;
using namespace std;

ImageProcess::ImageProcess(QObject *parent) : QThread(parent) {
    subWidget = new QWidget();
    ui.setupUi(subWidget);

    method = 0;

    subWidget->show();
    xbee = new Xbee("/dev/ttyUSB0");
}

ImageProcess::~ImageProcess() {
    delete this;
}

void ImageProcess::run() {

    qDebug() << "New thread started successfully!!";

}

void ImageProcess::sendXbee(char content){
    xbee->setDir(content);
    xbee->start();
}

void ImageProcess::processImage(cv::Mat &image) {
    cv::Mat process;
    vector<vector<Point> > squares;

    // Determine how to process the image
    process = edgeDetection(image);

    // Display processed image to another widget.
    QPixmap pix = QPixmap::fromImage(imageConvert(process));
    ui.displayLabel->setPixmap(pix);

    // Free the memory to avoid overflow.
    image.release();
}
// skin detection code!
bool R1(int R, int G, int B) {
    bool e1 = (R>95) && (G>40) && (B>20) && ((max(R,max(G,B)) - min(R, min(G,B)))>15) && (abs(R-G)>15) && (R>G) && (R>B);
    bool e2 = (R>220) && (G>210) && (B>170) && (abs(R-G)<=15) && (R>B) && (G>B);
    return (e1||e2);
}

bool R2(float Y, float Cr, float Cb) {
    bool e3 = Cr <= 1.5862*Cb+20;
    bool e4 = Cr >= 0.3448*Cb+76.2069;
    bool e5 = Cr >= -4.5652*Cb+234.5652;
    bool e6 = Cr <= -1.15*Cb+301.75;
    bool e7 = Cr <= -2.2857*Cb+432.85;
    return e3 && e4 && e5 && e6 && e7;
}

bool R3(float H, float S, float V) {
    return (H<25) || (H > 230);
}

cv::Mat ImageProcess::edgeDetection(cv::Mat &src) {
    // allocate the result matrix
        Mat dst = src.clone();
        int map[4][5];
        int rows=src.rows, cols=src.cols;
        //qDebug << "row"<<row;
        Vec3b cwhite = Vec3b::all(255);
        Vec3b cblack = Vec3b::all(0);

        Mat src_ycrcb, src_hsv;
        // OpenCV scales the YCrCb components, so that they
        // cover the whole value range of [0,255], so there's
        // no need to scale the values:
        cvtColor(src, src_ycrcb, CV_BGR2YCrCb);
        // OpenCV scales the Hue Channel to [0,180] for
        // 8bit images, so make sure we are operating on
        // the full spectrum from [0,360] by using floating
        // point precision:
        src.convertTo(src_hsv, CV_32FC3);
        cvtColor(src_hsv, src_hsv, CV_BGR2HSV);

        // Now scale the values between [0,255]:
        normalize(src_hsv, src_hsv, 0.0, 255.0, NORM_MINMAX, CV_32FC3);
        for(int i = 0; i< 4; i++){
            for(int j = 0; j< 5; j++){
                map[i][j] = 0;
            }
        }
        int leg[5];
        leg[0]=0;
        leg[1]=0;
        leg[2]=0;
        leg[3]=0;
        leg[4]=0;
        for(int i = 0; i < src.rows; i++) {
            for(int j = 0; j < src.cols; j++) {
                Vec3b pix_bgr = src.ptr<Vec3b>(i)[j];
                int B = pix_bgr.val[0];
                int G = pix_bgr.val[1];
                int R = pix_bgr.val[2];

                // apply rgb rule
                bool a = R1(R,G,B);

                Vec3b pix_ycrcb = src_ycrcb.ptr<Vec3b>(i)[j];
                int Y = pix_ycrcb.val[0];
                int Cr = pix_ycrcb.val[1];
                int Cb = pix_ycrcb.val[2];

                // apply ycrcb rule
                bool b = R2(Y,Cr,Cb);

                Vec3f pix_hsv = src_hsv.ptr<Vec3f>(i)[j];
                float H = pix_hsv.val[0];
                float S = pix_hsv.val[1];
                float V = pix_hsv.val[2];

                // apply hsv rule
                bool c = R3(H,S,V);

                if(!(a&&b&&c)){
                    dst.ptr<Vec3b>(i)[j] = cblack;
                }
                else{
                    //dst.ptr<Vec3b>(i)[j] = cwhite;
                    if(i<=rows/4 && j<=cols/5){
                        map[0][0]++;
                        leg[0]++;
                    }
                    else if(i<=rows/4 && j<=cols*2/5 && cols/5<=j){
                        map[0][1]++;
                        leg[1]++;
                    }
                    else if(i<=rows/4 && j<=cols*3/5 && cols*2/5<=j){
                        map[0][2]++;
                        leg[2]++;
                    }
                    else if(i<=rows/4 && j<=cols*4/5 && cols*3/5<=j){
                        map[0][3]++;
                        leg[3]++;
                    }
                    else if(i<=rows/4 && j<=cols && cols*4/5<=j){
                        map[0][4]++;
                        leg[4]++;
                    }
                    else if(rows/4<=i && i<=rows*2/4 &&  j<=cols/5){
                        map[1][0]++;
                        leg[0]++;
                    }
                    else if(rows/4<=i && i<=rows*2/4 && j<=cols*2/5 && cols/5<=j){
                        map[1][1]++;
                        leg[1]++;
                    }
                    else if(rows/4<=i && i<=rows*2/4 && j<=cols*3/5 && cols*2/5<=j){
                        map[1][2]++;
                        leg[2]++;
                    }
                    else if(rows/4<=i && i<=rows*2/4 && j<=cols*4/5 && cols*3/5<=j){
                        map[1][3]++;
                        leg[3]++;
                    }
                    else if(rows/4<=i && i<=rows*2/4 && j<=cols && cols*4/5<=j){
                        map[0][4]++;
                        leg[4]++;
                    }
                    else if(rows*2/4<=i && i<=rows*3/4 && j<=cols/4){
                        map[2][0]++;
                        leg[2]++;
                    }
                    else if(rows*2/4<=i && i<=rows*3/4 && j<=cols*2/4 && cols/4<=j){
                        map[2][1]++;
                        leg[2]++;
                    }
                    else if(rows*2/4<=i && i<=rows*3/4 && j<=cols*3/4 && cols*2/4<=j){
                        map[2][2]++;
                        leg[2]++;
                    }
                    else if(rows*2/4<=i && i<=rows*3/4 && j<=cols && cols*3/4<=j){
                        map[2][3]++;
                        leg[2]++;
                    }
                    else if(rows*3/4<=i && i<=rows && j<=cols/4){
                        map[3][0]++;
                        leg[3]++;
                    }
                    else if(rows*3/4<=i && i<=rows && j<=cols*2/4 && cols/4<=j){
                        map[3][1]++;
                        leg[3]++;
                    }
                    else if(rows*3/4<=i && i<=rows && j<=cols*3/4 && cols*2/4<=j){
                        map[3][2]++;
                        leg[3]++;
                    }
                    else if(rows*3/4<=i && i<=rows && j<=cols && cols*3/4<=j){
                        map[3][3]++;
                        leg[3]++;
                    }

                }
            }
        }
    //qDebug() <<map[0][0] << map[0][1] << map[0][2] << map[0][3] << map[0][4];
    int m1 = map[0][0], m2 = map[0][1], m3 = map[0][2], m4 = map[0][3], m5 = map[0][4];
    // turn left
    int max=0;
    for(int i=1; i<5; i++){
        if(max<map[0][i]){
            max = map[0][i];
        }
    }
    if(max>2000){
        if(m1+m2>m5+m4){
            qDebug() <<"Back-right";
            xbee->setDir('c');
            xbee->start();
        }
        else if(m5+m4>m1+m2){
            qDebug() <<"Back-left";
            xbee->setDir('z');
            xbee->start();
        }
        else{
            qDebug() <<"Back";
            xbee->setDir('x');
            xbee->start();
        }
    }
    else if(max>1300 && max <2000){
        if(m1+m2>m5+m4){
            qDebug() <<"right";
            xbee->setDir('d');
            xbee->start();
        }
        else if(m5+m4>m1+m2){
            qDebug() <<"left";
            xbee->setDir('a');
            xbee->start();
        }
        else{
            qDebug() <<"Fine";
            xbee->setDir('s');
            xbee->start();
        }
    }
    else{
        if(m1+m2>m5+m4){
            qDebug() <<"Front-right";
            xbee->setDir('e');
            xbee->start();
        }
        else if(m5+m4>m1+m2){
            qDebug() <<"Front-left";
            xbee->setDir('q');
            xbee->start();
        }
        else{
            qDebug() <<"Front";
            xbee->setDir('w');
            xbee->start();
        }
    }


    /*
    if(m1+m2+m3>m3+m4+m5){
        qDebug() << "R";
        xbee->setDir('r');
        xbee->start();
    }
    else if(abs(m2+m1-m4-m5)<1000){
        qDebug() << "C";
       xbee->setDir('c');
        xbee->start();
    }
    else if(m1+m2+m3<m3+m4+m5){
        qDebug() << "L";
        xbee->setDir('l');
        xbee->start();
    }
    */
    return dst;

}

// edge detection code!
/*
cv::Mat ImageProcess::edgeDetection(cv::Mat &image) {
    int houghVote = 200;
    cvtColor(image,image,CV_RGB2GRAY);

    // Canny algorithm
    Mat contours;
    Canny(image,contours,25,250);
    Mat contoursInv;
    threshold(contours,contoursInv,25,255,THRESH_BINARY_INV);

    // Lines detection
    std::vector<Vec2f> lines;
    if (houghVote < 1 or lines.size() > 2){ // we lost all lines. reset
        houghVote = 300;
    }
    else{ houghVote += 25;}
    while(lines.size() < 5 && houghVote > 0){
        HoughLines(contours,lines,1,PI/180, houghVote);
        houghVote -= 5;
    }
    Mat result(image.size(),CV_8U,Scalar(255));
    image.copyTo(result);

    // Draw the lines
    std::vector<Vec2f>::const_iterator it= lines.begin();
    //Mat edge_mat(image.size(),CV_8U,Scalar(0, 0, 255));
    //edge_mat = Mat();
    Mat edge_mat;
    edge_mat = Mat::zeros(image.size(),CV_8U);
    //edge_mat.clear;
    float bottom_avg = 0;
    int middle = image.rows/2.0;
    int rows = image.rows;
    int cols = image.cols;
    int i=0, j=edge_mat.rows-1;
    int bottom_cnt = 0;
    while (it!=lines.end()) {
        float rho= (*it)[0];
        float theta= (*it)[1];
        if( theta > PI*3.0/4.0 | theta < PI/4.0){
            Point pt1(rho/cos(theta),0);
            Point pt2(rho/cos(theta)-result.rows*sin(theta)/cos(theta),result.rows);
            float top = rho/cos(theta);
            float bottom = rho/cos(theta)-result.rows*sin(theta)/cos(theta);
            //if(top < 220 && top > 100){
                bottom_avg += bottom;
                bottom_cnt++;
                line(edge_mat, pt1, pt2, Scalar(255, 0, 0), 3, 8);
                line(result, pt1, pt2, Scalar(0, 0, 255), 3, 8);
            //}
        }
        ++it;
    }
    bottom_avg=bottom_avg/bottom_cnt;

    if(bottom_avg >= 140 && bottom_avg <= 180){
        qDebug() << "C";
        xbee->setDir('c');
        xbee->start();
    }
    else if(bottom_avg < 140){
        qDebug() << "L";
        xbee->setDir('l');
        xbee->start();
    }
    else if(bottom_avg > 180){
        qDebug() << "R";
        xbee->setDir('r');
        xbee->start();
    }
    else{
        qDebug() << "Err";
        xbee->setDir('e');
        xbee->start();
    }

    return edge_mat;
    //return result;
}
*/
void ImageProcess::findSquares(cv::Mat &image, vector<vector<Point> > &squares) {
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;
    int thresh = 50, N = 11;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        cv::mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                cv::Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                cv::dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                // tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            cv::findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                cv::approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                        fabs(contourArea(Mat(approx))) > 1000 &&
                        cv::isContourConvex(Mat(approx)) )
                {
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 )
                        squares.push_back(approx);
                }
            }
        }
    }
}

void ImageProcess::drawSquares(cv::Mat &image, vector<vector<Point> > &squares) {
    // the function draws all the squares in the image
    for( size_t i = 0; i < squares.size(); i++ ) {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();

        polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, CV_AA);
    }
}

cv::Mat ImageProcess::findCircles(cv::Mat &image) {
    cv::Mat circle;
    vector<Vec3f> circles;

    // Convert it to gray
    cvtColor(image, circle, CV_BGR2GRAY);

    // Reduce the noise so we avoid false circle detection
    GaussianBlur(circle, circle, Size(9, 9), 2, 2);

    // Apply the Hough Transform to find the circles
    HoughCircles(circle, circles, CV_HOUGH_GRADIENT, 1, circle.rows/16, 150, 30, 0, 0);

    // Draw the circles detected
    for(size_t i = 0; i < circles.size(); i++ ) {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        // circle center
        cv::circle(image, center, 3, cv::Scalar(0,255,0), -1, 8, 0 );
        // circle outline
        cv::circle(image, center, radius, cv::Scalar(0,0,255), 3, 8, 0 );
    }

    return image;
}

cv::Mat ImageProcess::findLines(cv::Mat &image) {
    cv::Mat line;
    vector<Vec4i> lines;

    // Convert it to gray
    cvtColor(image, line, CV_BGR2GRAY);

    // Detect the edges of the image by using a Canny detector
    Canny(line, line, 50, 200, 3);
    //qDebug() << "line";

    // Apply the Hough Transform to find the line segments
    HoughLinesP(line, lines, 1, CV_PI/180, 50, 50, 10);

    // Draw the line segments detected
    for(size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];

        cv::line(image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
    }

    return image;
}

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
double ImageProcess::angle(Point pt1, Point pt2, Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// The following slots are used to set image processing mode.
void ImageProcess::changeToNormal() {
    method = 1;
}

void ImageProcess::changeToEdgeDetection() {
    method = 1;
}

void ImageProcess::changeToCircleDetection() {
    method = 1;
}

void ImageProcess::changeToSquareDetection() {
    method = 1;
}

void ImageProcess::changeToLineDetection() {
    method = 1;
}

QImage ImageProcess::imageConvert(cv::Mat &matImage) {
    QImage::Format format;

    // If we use edge detection, we will use gray scale to display image.
    switch(method) {
    case 1: format = QImage::Format_Indexed8;
        break;
    default:format = QImage::Format_RGB888;
        break;
    }

    // Convert processed openCV frame to Qt's image format in order to display.
    QImage qImage(
                (uchar*)matImage.data,
                matImage.cols,
                matImage.rows,
                matImage.step,
                format
                );

    return qImage.rgbSwapped().mirrored(false, false);
}
