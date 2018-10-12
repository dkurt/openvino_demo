#include <opencv2/opencv.hpp>

#include "runners.hpp"

using namespace cv;
using namespace cv::dnn;

OCVRunner::OCVRunner(const std::string& modelName, const std::string& target) : Runner(modelName, target) {
  std::cout << modelName << " using OpenCV runner" << std::endl;

  net = readNet(xmlPath, binPath);
  if (device == "CPU")
    net.setPreferableTarget(DNN_TARGET_CPU);
  else if (device == "GPU")
  {
    if (precision == "FP32")
      net.setPreferableTarget(DNN_TARGET_OPENCL);
    else if (precision == "FP16")
      net.setPreferableTarget(DNN_TARGET_OPENCL_FP16);
    else
      CV_Error(Error::StsNotImplemented, "Unknown precision: " + precision);
  }
  else if (device == "MYRIAD")
    net.setPreferableTarget(DNN_TARGET_MYRIAD);
  else
    CV_Error(Error::StsNotImplemented, "Unknown device: " + device);
}

void OCVRunner::run(const cv::Mat& input, cv::Mat& output, cv::TickMeter& tm) {
  net.setInput(input);
  tm.reset();
  tm.start();
  output = net.forward();
  tm.stop();
}
