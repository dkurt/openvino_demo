#include <opencv2/opencv.hpp>

#include "runners.hpp"

using namespace cv;
using namespace cv::dnn;

OCVRunner::OCVRunner(const std::string& modelName, int target) : Runner(modelName, target) {
  std::cout << modelName << " using OpenCV runner" << std::endl;

  net = readNet(xmlPath, binPath);
  if (target == GPU_FP32)
    net.setPreferableTarget(DNN_TARGET_OPENCL);
  else if (target == GPU_FP16)
    net.setPreferableTarget(DNN_TARGET_OPENCL_FP16);
  else if (target == MYRIAD)
    net.setPreferableTarget(DNN_TARGET_MYRIAD);
}

void OCVRunner::run(const cv::Mat& input, cv::Mat& output, cv::TickMeter& tm) {
  net.setInput(input);
  tm.reset();
  tm.start();
  output = net.forward();
  tm.stop();
}
