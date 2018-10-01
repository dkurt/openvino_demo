#include <inference_engine.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

class Runner {
public:
  Runner(const std::string& modelName)
  {
    std::string prefix = cv::utils::fs::join(getenv("INTEL_CVSDK_DIR"),
                         cv::utils::fs::join("deployment_tools",
                         cv::utils::fs::join("intel_models",
                         cv::utils::fs::join(modelName,
                         cv::utils::fs::join("FP32", modelName)))));
    xmlPath = prefix + ".xml";
    binPath = prefix + ".bin";
  }

  virtual void run(const cv::Mat& input, cv::Mat& output) = 0;

protected:
  std::string xmlPath;
  std::string binPath;
};

// Intel's Inference Engine runner.
class IERunner : public Runner {
public:
  IERunner(const std::string& modelName);

  virtual void run(const cv::Mat& input, cv::Mat& output);

private:
  InferenceEngine::CNNNetwork net;
  InferenceEngine::BlobMap inputBlobs, outputBlobs;
  InferenceEngine::InferRequest infRequest;
};

// OpenCV runner.
class OCVRunner : public Runner {
public:
  OCVRunner(const std::string& modelName);

  virtual void run(const cv::Mat& input, cv::Mat& output);

private:
  cv::dnn::Net net;
};
