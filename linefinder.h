/*------------------------------------------------------------------------------------------*\
   Lane Detection

   General idea and some code modified from:
   chapter 7 of Computer Vision Programming using the OpenCV Library. 
   by Robert Laganiere, Packt Publishing, 2011.

   This program is free software; permission is hereby granted to use, copy, modify, 
   and distribute this source code, or portions thereof, for any purpose, without fee, 
   subject to the restriction that the copyright notice may not be removed 
   or altered from any source or altered source distribution. 
   The software is released on an as-is basis and without any warranties of any kind. 
   In particular, the software is not guaranteed to be fault-tolerant or free from failure. 
   The author disclaims all warranties with regard to this software, any use, 
   and any consequent failure, is purely the responsibility of the user.
 
   Copyright (C) 2013 Jason Dorweiler, www.transistor.io
\*------------------------------------------------------------------------------------------*/

#if !defined LINEF
#define LINEF

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#define PI 3.1415926
#include <string>

class LineFinder {

  private:

	  // original image
	  cv::Mat img;

	  // vector containing the end points 
	  // of the detected lines
	  std::vector<cv::Vec4i> lines;

	  // accumulator resolution parameters
	  double deltaRho;
	  double deltaTheta;

	  // minimum number of votes that a line 
	  // must receive before being considered
	  int minVote;

	  // min length for a line
	  double minLength;

	  // max allowed gap along the line
	  double maxGap;

	  // distance to shift the drawn lines down when using a ROI
	  int shift;

  public:

	  // Default accumulator resolution is 1 pixel by 1 degree
	  // no gap, no mimimum length
	  LineFinder() : deltaRho(1), deltaTheta(PI/180), minVote(10), minLength(0.), maxGap(0.) {}

	  // Set the resolution of the accumulator
	  void setAccResolution(double dRho, double dTheta) {

		  deltaRho= dRho;
		  deltaTheta= dTheta;
	  }

	  // Set the minimum number of votes
	  void setMinVote(int minv) {

		  minVote= minv;
	  }

	  // Set line length and gap
	  void setLineLengthAndGap(double length, double gap) {

		  minLength= length;
		  maxGap= gap;
	  }

	  // set image shift
	  void setShift(int imgShift) {

		  shift = imgShift;
	  }

	  // Apply probabilistic Hough Transform
	  std::vector<cv::Vec4i> findLines(cv::Mat& binary) {

		  lines.clear();
		  cv::HoughLinesP(binary,lines,deltaRho,deltaTheta,minVote, minLength, maxGap);

		  return lines;
	  }

	  // Draw the detected lines on an image
	  void drawDetectedLines(cv::Mat &image, cv::Scalar color=cv::Scalar(255)) {
		
		  // Draw the lines
		  std::vector<cv::Vec4i>::const_iterator it2= lines.begin();
			std::vector<cv::Point> dat;

		std::stringstream stream;

			std::string str="lines: ";
		  while (it2!=lines.end()) {
		
			  cv::Point pt1((*it2)[0],(*it2)[1]+shift);        
			  cv::Point pt2((*it2)[2],(*it2)[3]+shift);
			if(dat.size()>0&& image.cols-10>pt1.x &&10<pt1.x &&image.cols-10>pt2.x && 10<pt2.x){

				// std::cout << " x: ("<<dat[0].x-pt1.x<<")\n";
				if(abs(dat[0].x-pt1.x)>30 && abs(dat[0].x-pt1.x)<70){
					//	cv::line( image, dat[0], pt1, color, 6 );
						int x=(dat[0].x+pt1.x)/2; int y=(dat[0].y+pt1.y)/2;
						cv::circle(image,cv::Point(x,y),3,cv::Scalar(0,255,0),10);
				
			//stream<<lines.size();
			//str+=stream.str()+"\n";
			//str+=" x: "+(dat[0].x+pt1.x)/2;
			//str+=" y: "+(dat[0].y+pt1.y)/2;
			std::cout << " x: ("<<(dat[0].x+pt1.x)/2<<")\n";
			std::cout << " y: ("<<(dat[0].y+pt1.y)/2<<")\n";
			
			//putText(image, str, cv::Point(10,image.rows-100), 2, 0.8, cv::Scalar(0,255,0),0);
					}
				
				dat.clear();
				}
			
			 dat.push_back(cv::Point(pt1));
			if(image.cols-10>pt1.x &&10<pt1.x &&image.cols-10>pt2.x && 10<pt2.x){
			  cv::line( image, pt1, pt2, color, 6 );
			}
			 std::cout << " HoughP line: ("<< pt1 <<"," << pt2 << ")\n";

			
			//cv::rectangle(image,dat, dat2,cv::Scalar(0,255,0));
			

			

			
				//break;
			  ++it2;	
		  }
			dat.clear();
	  }

	  // Eliminates lines that do not have an orientation equals to
	  // the ones specified in the input matrix of orientations
	  // At least the given percentage of pixels on the line must 
	  // be within plus or minus delta of the corresponding orientation
	  std::vector<cv::Vec4i> removeLinesOfInconsistentOrientations(
		  const cv::Mat &orientations, double percentage, double delta) {

			  std::vector<cv::Vec4i>::iterator it= lines.begin();
	
			  // check all lines
			  while (it!=lines.end()) {

				  // end points
				  int x1= (*it)[0];
				  int y1= (*it)[1];
				  int x2= (*it)[2];
				  int y2= (*it)[3];
		   			
				  // line orientation + 90o to get the parallel line
				  double ori1= atan2(static_cast<double>(y1-y2),static_cast<double>(x1-x2))+PI/2;
				  if (ori1>PI) ori1= ori1-2*PI;

				  double ori2= atan2(static_cast<double>(y2-y1),static_cast<double>(x2-x1))+PI/2;
				  if (ori2>PI) ori2= ori2-2*PI;
	
				  // for all points on the line
				  cv::LineIterator lit(orientations,cv::Point(x1,y1),cv::Point(x2,y2));
				  int i,count=0;
				  for(i = 0, count=0; i < lit.count; i++, ++lit) { 
		
					  float ori= *(reinterpret_cast<float *>(*lit));

					  // is line orientation similar to gradient orientation ?
					  if (std::min(fabs(ori-ori1),fabs(ori-ori2))<delta)
						  count++;
		
				  }

				  double consistency= count/static_cast<double>(i);

				  // set to zero lines of inconsistent orientation
				  if (consistency < percentage) {
 
					  (*it)[0]=(*it)[1]=(*it)[2]=(*it)[3]=0;

				  }

				  ++it;
			  }

			  return lines;
	  }
};


#endif
