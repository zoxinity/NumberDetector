#include "DrawText.hpp"

void drawText(cv::Mat &image)
{
	putText(image, "Hello OpenCV",
			cv::Point(20, 50),
			cv::FONT_HERSHEY_COMPLEX, 1, // font face and scale
			cv::Scalar(255, 255, 255), // white
			1, cv::LINE_AA); // line thickness and type
}