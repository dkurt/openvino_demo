#include <opencv2/opencv.hpp>

#include "runners.hpp"

using namespace cv;
using namespace cv::dnn;

OCVRunner::OCVRunner(const std::string& modelName) : Runner(modelName) {
  std::cout << modelName << " using OpenCV runner" << std::endl;
  net = readNet(xmlPath, binPath);
}

void OCVRunner::run(const cv::Mat& input, cv::Mat& output) {
  net.setInput(input);
  output = net.forward();
}
