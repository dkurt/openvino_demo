#include <opencv2/opencv.hpp>

#include "runners.hpp"

const char* kWinName = "Intel OpenVINO toolkit";
int fdTarget = CPU;
int erTarget = MYRIAD;
int frTarget = CPU;

const char* keys =
    "{ help    h | | Print help message. }"
    "{ gallery g | | A path to gallery with images. }";

class Algorithm {
public:
  Algorithm(cv::Ptr<Runner> runner_) : runner(runner_) {}

protected:
  cv::Ptr<Runner> runner;
};

// This class runs a network which detects faces.
class FaceDetector : public Algorithm {
public:
  FaceDetector() : Algorithm(new IERunner("face-detection-retail-0004", fdTarget)) {}

  // Returns bounding boxes around predicted faces.
  void detect(const cv::Mat& frame, std::vector<cv::Rect>& boxes, cv::TickMeter& tm,
              float confThr = 0.7) {
    boxes.clear();

    cv::Mat detections;
    cv::Mat blob = cv::dnn::blobFromImage(frame, /*scale*/1.0, cv::Size(300, 300),
                                          /*mean*/cv::Scalar(), /*swapRB*/false,
                                          /*crop*/false);
    runner->run(blob, detections, tm);

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
        int left   = std::max(0, std::min((int)(detections.at<float>(i, 3) * frame.cols), frame.cols - 1));
        int top    = std::max(0, std::min((int)(detections.at<float>(i, 4) * frame.rows), frame.rows - 1));
        int right  = std::max(0, std::min((int)(detections.at<float>(i, 5) * frame.cols), frame.cols - 1));
        int bottom = std::max(0, std::min((int)(detections.at<float>(i, 6) * frame.rows), frame.rows - 1));
        int width  = right - left + 1;
        int height = bottom - top + 1;
        boxes.push_back(cv::Rect(left, top, width, height));
      }
    }
  }
};

class EmotionsRecognizer : public Algorithm {
public:
  EmotionsRecognizer() : Algorithm(new OCVRunner("emotions-recognition-retail-0003", erTarget)) {}

  // Returns a string with face's emotion.
  std::string recognize(const cv::Mat& faceImg, cv::TickMeter& tm) {
    cv::Mat confidences;
    cv::Mat blob = cv::dnn::blobFromImage(faceImg, /*scale*/1.0, cv::Size(64, 64),
                                          /*mean*/cv::Scalar(), /*swapRB*/false,
                                          /*crop*/false);
    runner->run(blob, confidences, tm);

    // Emotions recognition network is a classification model which predicts
    // a vector of 5 confidences corresponding to face emotions: neutral, happy,
    // sad, surprised, angry. Output shape is 1x1x5x1
    confidences = confidences.reshape(1, 1);  // Reshape to a single row.

    double maxConf;
    cv::Point maxLoc;
    minMaxLoc(confidences, 0, &maxConf, 0, &maxLoc);
    if (maxLoc.x == 0) return "neutral";
    else if (maxLoc.x == 1) return "happy";
    else if (maxLoc.x == 2) return "sad";
    else if (maxLoc.x == 3) return "surprised";
    else return "angry";
  }
};

// NOTE: do not run this model using OCVRunner due a bug in R3 release
//       (will be fixed in the future releases)
class FaceRecognition : public Algorithm {
public:
  FaceRecognition(const std::string& gallery, FaceDetector& fd)
      : Algorithm(new IERunner("face-reidentification-retail-0001", frTarget)) {
    cv::TickMeter tm;
    if (cv::utils::fs::isDirectory(gallery))
    {
      // Get all the images from gallery.
      std::vector<cv::String> images;
      cv::utils::fs::glob_relative(gallery, "", images, false, true);
      for (const std::string& imgName : images) {
        std::string imgPath = cv::utils::fs::join(gallery, imgName);
        std::string personName = imgName.substr(0, imgName.rfind('.'));
        cv::Mat img = cv::imread(imgPath);

        // Detect a face on image.
        std::vector<cv::Rect> boxes;
        fd.detect(img, boxes, tm);

        if (boxes.empty())
            CV_Error(cv::Error::StsAssert, "There is no face found on image " + imgName);
        if (boxes.size() > 1)
            CV_Error(cv::Error::StsAssert, "More than one face found on image " + imgName);

        cv::Mat face = img(boxes[0]);
        cv::Mat embedding = getEmbedding(face, tm);
        embeddings[personName] = embedding;
      }
    }
    std::cout << "Number of registered persons: " << embeddings.size() << std::endl;
  }

  // Returns a name of person from gallery if matching score is higher than threshold value.
  // If no person found, returns 'unknown'.
  // Thresholding score is in range [-1, 1] where 1 means 100% matching.
  std::string recognize(const cv::Mat& faceImg, cv::TickMeter& tm, float scoreThr = 0.5) {
    cv::Mat embedding = getEmbedding(faceImg, tm);
    std::string bestMatchName = "unknown";
    for (auto& it : embeddings) {
      float score = embedding.dot(it.second);
      if (score > scoreThr) {
        bestMatchName = it.first;
        scoreThr = score;
      }
    }
    return bestMatchName;
  }

private:
  std::map<std::string, cv::Mat> embeddings;

  // Returns an embedding vector of 128 floating point values.
  cv::Mat getEmbedding(const cv::Mat& faceImg, cv::TickMeter& tm) {
    cv::Mat embedding;
    cv::Mat blob = cv::dnn::blobFromImage(faceImg, /*scale*/1.0, cv::Size(128, 128),
                                          /*mean*/cv::Scalar(), /*swapRB*/false,
                                          /*crop*/false);
    runner->run(blob, embedding, tm);

    embedding = embedding.reshape(1, 1);  // Reshape from 1x1x128x1 to 1x128
    cv::normalize(embedding, embedding);  // Make a unit vector (L2 norm).
    return embedding;
  }
};

static void drawBox(cv::Mat& frame, const cv::Rect& box, int& thickness) {
  int size = std::min(box.width, box.height);
  thickness = 0.05 * size;
  size *= 0.35;
  cv::Scalar colors[] = {cv::Scalar(0, 0, 0), cv::Scalar(217, 101, 23)};

  for (int i = 0; i < 2; ++i)
  {
    cv::Point shift((1 - i) * 0.5 * thickness, (1 - i) * 0.5 * thickness);
    cv::line(frame, box.tl() + shift, cv::Point(box.x + size, box.y) + shift, colors[i], thickness);
    cv::line(frame, box.tl() + shift, cv::Point(box.x, box.y + size) + shift, colors[i], thickness);

    cv::line(frame, box.br() + shift, cv::Point(box.x + box.width - size, box.y + box.height) + shift, colors[i], thickness);
    cv::line(frame, box.br() + shift, cv::Point(box.x + box.width, box.y + box.height - size) + shift, colors[i], thickness);

    cv::Point corner(box.x, box.y + box.height);
    cv::line(frame, corner + shift, cv::Point(corner.x + size, corner.y) + shift, colors[i], thickness);
    cv::line(frame, corner + shift, cv::Point(corner.x, corner.y - size) + shift, colors[i], thickness);

    corner = cv::Point(box.x + box.width, box.y);
    cv::line(frame, corner + shift, cv::Point(corner.x - size, corner.y) + shift, colors[i], thickness);
    cv::line(frame, corner + shift, cv::Point(corner.x, corner.y + size) + shift, colors[i], thickness);
  }
}

static const char* targetToStr(int target) {
  if (target == CPU) return "CPU";
  if (target == GPU_FP32) return "GPU, FP32";
  if (target == GPU_FP16) return "GPU, FP16";
  if (target == MYRIAD) return "VPU";
  return "";
}

int main(int argc, char** argv) {
  // Parse command line arguments to get a path to gallery.
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Intel OpenVINO demonstration application.");
  if (parser.has("help"))
  {
      parser.printMessage();
      return 0;
  }

  std::string gallery = parser.get<std::string>("gallery");

  cv::VideoCapture cap(0);
  cv::Mat frame;

  FaceDetector fd;
  EmotionsRecognizer er;
  FaceRecognition fr(gallery, fd);

  cv::namedWindow(kWinName, cv::WINDOW_NORMAL);

  std::string logoPath = cv::utils::fs::join(getenv("INTEL_CVSDK_DIR"),
                         cv::utils::fs::join("deployment_tools",
                         cv::utils::fs::join("computer_vision_algorithms",
                         cv::utils::fs::join("share",
                         cv::utils::fs::join("cva",
                         cv::utils::fs::join("FaceDetection",
                         cv::utils::fs::join("doc",
                         cv::utils::fs::join("html", "intellogo.png"))))))));
  cv::Mat logo = cv::imread(logoPath);
  cv::Rect logoRoi(0, 0, logo.cols, logo.rows);
  cv::Mat logoMask;
  cv::inRange(logo, cv::Scalar(), cv::Scalar(), logoMask);
  logoMask = ~logoMask;

  cv::TickMeter timers[3];

  while (cv::waitKey(1) < 0) {
    // Capture a frame from a camera.
    cap >> frame;
    if (frame.empty())
      break;

    std::vector<cv::Rect> boxes;
    fd.detect(frame, boxes, timers[0]);

    std::vector<std::string> emotions;
    std::vector<std::string> names;
    for (auto& box : boxes) {
      cv::Mat face = frame(box);
      emotions.push_back(er.recognize(face, timers[1]));
      names.push_back(fr.recognize(face, timers[2]));
    }

    for (int i = 0; i < boxes.size(); ++i) {
      const cv::Rect& box = boxes[i];

      // Draw a bounding box around a face.
      int boxThickness;
      drawBox(frame, box, boxThickness);

      // Draw a label.
      std::string label = names[i] + ", " + emotions[i];

      float fontScale = 0.005 * box.width;

      int baseLine;
      cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, &baseLine);

      int top = std::max(box.y - boxThickness, labelSize.height);
      cv::putText(frame, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 255, 0), 2);
    }

    // Put logo image.
    logo.copyTo(frame(logoRoi), logoMask);

    // Put efficiency information.
    int targets[] = {fdTarget, erTarget, frTarget};
    for (int i = 0; i < 3; ++i) {
      const char* algo = i == 0 ? "DETECTION  " : (i == 1 ? "EMOTIONS   " : "RECOGNITION");
      std::string label = cv::format("%s (%s): %.1f FPS", algo, targetToStr(targets[i]),
                                     1.0 / timers[i].getTimeSec());

      float fontScale = 0.001 * frame.cols;
      int baseLine;
      cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, &baseLine);
      cv::putText(frame, label, cv::Point(20, frame.rows - 1.5 * (3 - i) * labelSize.height), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow(kWinName, frame);
  }
  return 0;
}
