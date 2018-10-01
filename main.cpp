#include <opencv2/opencv.hpp>

#include "runners.hpp"

const char* kWinName = "Intel OpenVINO toolkit";

class Algorithm {
public:
  Algorithm(cv::Ptr<Runner> runner_) : runner(runner_) {}

protected:
  cv::Ptr<Runner> runner;
};

// This class runs a network which detects faces.
class FaceDetector : public Algorithm {
public:
  FaceDetector() : Algorithm(new IERunner("face-detection-retail-0004")) {}

  // Returns bounding boxes around predicted faces.
  void detect(const cv::Mat& frame, std::vector<cv::Rect>& boxes, float confThr = 0.5) {
    boxes.clear();

    cv::Mat detections;
    cv::Mat blob = cv::dnn::blobFromImage(frame, /*scale*/1.0, cv::Size(300, 300),
                                          /*mean*/cv::Scalar(), /*swapRB*/false,
                                          /*crop*/false);
    runner->run(blob, detections);

    // Output of SSD-based object detection network is a 4D blob with shape 1x1xNx7
    // where N is a number of detections and an every detection is a vector
    // [batchId, classId, confidence, left, top, right, bottom].

    detections = detections.reshape(1, detections.total() / 7);  // Reshape from 4D to 2D

    const int numDetections = detections.rows;
    for (int i = 0; i < numDetections; ++i) {
      float confidence = detections.at<float>(i, 2);
      // Exclude predictions with low confidence.
      if (confidence > confThr) {
        // Predicted bounding boxes have relative coordinates in range [0, 1].
        int left   = (int)(detections.at<float>(i, 3) * frame.cols);
        int top    = (int)(detections.at<float>(i, 4) * frame.rows);
        int right  = (int)(detections.at<float>(i, 5) * frame.cols);
        int bottom = (int)(detections.at<float>(i, 6) * frame.rows);
        int width  = right - left + 1;
        int height = bottom - top + 1;
        boxes.push_back(cv::Rect(left, top, width, height));
      }
    }
  }
};


int main(int argc, char** argv) {
  cv::VideoCapture cap(0);
  cv::Mat frame;

  FaceDetector fd;

  cv::namedWindow(kWinName, cv::WINDOW_NORMAL);

  while (cv::waitKey(1) < 0) {
    // Capture a frame from a camera.
    cap >> frame;
    if (frame.empty())
      break;

    std::vector<cv::Rect> boxes;
    fd.detect(frame, boxes);

    for (auto& box : boxes) {
      cv::rectangle(frame, box, cv::Scalar(0, 255, 0));
    }

    cv::imshow(kWinName, frame);
  }

  return 0;
}
