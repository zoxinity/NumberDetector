#include "DrawText.hpp"

int main()
{
	std::cout << "Built with OpenCV " << CV_VERSION << std::endl;
	cv::Mat image;
	cv::VideoCapture capture;
	capture.open(0);
	if(capture.isOpened())
	{
		std::cout << "Capture is opened" << std::endl;
		for(;;)
		{
			capture >> image;
			if(image.empty())
				break;
			drawText(image);
			imshow("Sample", image);
			if(cv::waitKey(10) >= 0)
				break;
		}
	}
	else
	{
		std::cout << "No capture" << std::endl;
		image = cv::Mat::zeros(480, 640, CV_8UC1);
		drawText(image);
		cv::imshow("Sample", image);
		cv::waitKey(0);
	}
	return 0;
}