

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include "linefinder.h"

#define PI 3.1415926
int ratio = 3;
int kernel_size = 3;
int lowThreshold=160;
//videoda islencek alani belirleme (videoyu kesme deyebliriz)
int roiX=0,roiX_width=550;
int roiY=10,roiX_height=200;

using namespace cv;

int main(int argc, char* argv[]) {
	int houghVote = 200;
	string arg = argv[1];
	bool showSteps = argv[2];

	string window_name = "Processed Video";
	//namedWindow(window_name, WINDOW_AUTOSIZE); //resizable window;
	VideoCapture capture(arg);

	if (!capture.isOpened()) //if this fails, try to open as a video camera, through the use of an integer param
        	{capture.open(atoi(arg.c_str()));}

	double dWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

	std::cout << "Frame Size = " << dWidth << "x" << dHeight << std::endl;

	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));

	VideoWriter oVideoWriter ("LaneDetectionfd.avi", CV_FOURCC('P','I','M','1'), 20, frameSize, true); //initialize the VideoWriter object 

        Mat image;
	image = imread(argv[1]);	
        while (1)
        {
           	capture >> image;
            if (image.empty())
                break;
            	Mat gray;
            	cvtColor(image,gray,CV_RGB2GRAY);
            	vector<string> codes;
            	Mat corners;
            	findDataMatrix(gray, codes, corners);
            	drawDataMatrixCodes(image, codes, corners); 
		//std::cout <<" Size = " << image.cols/5 << " W " << image.cols-1<< " H " <<image.rows-image.cols/5 << std::endl;
		//Rect roi(0,image.cols/3,image.cols-1,image.rows-image.cols/3);// set the ROI for the image
		//Rect roi(roiX,120,image.cols-1,110);
		//Mat imgROI =image(roi);
		Mat imgROI =image;
    // Display the image
	if(showSteps){
	//	imshow("Original Image",imgROI);
	//	waitKey(0);
	}

   // Canny algorithm
	Mat contours;
	//Canny(imgROI,contours,50,250);

	Canny( imgROI, contours, lowThreshold, lowThreshold*ratio, kernel_size );
	Mat contoursInv;
	threshold(contours,contoursInv,128,255,THRESH_BINARY_INV|CV_THRESH_OTSU);

   // Display Canny image
	if(showSteps){
		imshow("Contours1",contoursInv);
	//	waitKey(0);
	}

 /* 
	Hough tranform for line detection with feedback
	Increase by 25 for the next frame if we found some lines.  
	This is so we don't miss other lines that may crop up in the next frame
	but at the same time we don't want to start the feed back loop from scratch. 
*/
	std::vector<Vec2f> lines;
	if (houghVote < 1 or lines.size() > 2){ // we lost all lines. reset 
		houghVote = 200; 
	}
	else{ houghVote += 25;} 
	while(lines.size() < 5 && houghVote > 0){
		HoughLines(contours,lines,1,PI/180, houghVote);
		houghVote -= 5;
	}
	std::cout << houghVote << "\n";
	Mat result(imgROI.size(),CV_8U,Scalar(255));
	imgROI.copyTo(result);

   // Draw the limes
	std::vector<Vec2f>::const_iterator it= lines.begin();
	Mat hough(imgROI.size(),CV_8U,Scalar(0));
	std::cout<<"column: "<<result.cols<< " rows: "<<result.rows;
	while (it!=lines.end()) {

		//result.cols  sutun sayisi     
		//result.rows satir sayisini verir
		// orn. c1.mp4 de 200 sutun 348 satir var
		
		float rho= (*it)[0];   // first element is distance rho
		float theta= (*it)[1]; // second element is angle theta
		
		if ( theta > 0.09 && theta < 1.48 || theta < 3.14 && theta > 1.66 ) { // filter to remove vertical and horizontal lines
			//if(result.cols>rho && 1<rho){
			// point of intersection of the line with first row
			Point pt1(rho/cos(theta),0);        
			// point of intersection of the line with last row
			Point pt2((rho-result.rows*sin(theta))/cos(theta),result.rows);
			// draw a white line
			line( result, pt1, pt2, Scalar(255), 8); 
			line( hough, pt1, pt2, Scalar(255), 8);
			//}
		}

		//std::cout << "line: (" << rho << "," << theta << ")\n"; 
		++it;
	}

    // Display the detected line image
	if(showSteps){
		imshow("Detected Lines with Hough",result);
	//	waitKey(0);
	}
   // Create LineFinder instance
	LineFinder ld;

   // Set probabilistic Hough parameters
	ld.setLineLengthAndGap(60,10);
	ld.setMinVote(4);

   // Detect lines
	std::vector<Vec4i> li= ld.findLines(contours);
	Mat houghP(imgROI.size(),CV_8U,Scalar(0));
	ld.setShift(0);
	ld.drawDetectedLines(houghP);
	std::cout << "First Hough" << "\n";

	if(showSteps){
		imshow("Detected Lines with HoughP", houghP);
	//	waitKey(0);
	}

   // bitwise AND of the two hough images
	bitwise_and(houghP,hough,houghP);
	Mat houghPinv(imgROI.size(),CV_8U,Scalar(0));
	Mat dst(imgROI.size(),CV_8U,Scalar(0));
	threshold(houghP,houghPinv,150,255,THRESH_BINARY_INV); // threshold and invert to black lines

	if(showSteps){
	//	imshow("Detected Lines with Bitwise", houghPinv);
	//	waitKey(0);
	}

	Canny(houghPinv,contours,100,350);
	li= ld.findLines(contours);
   // Display Canny image
	if(showSteps){
		imshow("Contours2",contours);
	//	waitKey(0);
	}

	   // Set probabilistic Hough parameters
	ld.setLineLengthAndGap(1,1);
	ld.setMinVote(1);
	//ld.setShift(120);//ld.setShift(roiY);//image.cols/3);
	ld.drawDetectedLines(image);
		
	std::stringstream stream;
	stream << "Lines Segments: " << lines.size();
	
	putText(image, stream.str(), Point(10,image.rows-10), 2, 0.8, Scalar(0,0,255),0);

        imshow(window_name, image); 
	//waitKey(0);

	oVideoWriter.write(image); //writer the frame into the file

	char key = (char) waitKey(10);
	lines.clear();
	}
}




