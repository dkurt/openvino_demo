#include <inference_engine.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

class Runner {
public:
  Runner(const std::string& modelName, std::string target)
  {
    std::transform(target.begin(), target.end(), target.begin(), ::toupper);
    size_t delim = target.find('_');
    device = target.substr(0, delim);
    if (delim != std::string::npos)
        precision = target.substr(delim + 1);
    else
        precision = target == "MYRIAD" ? "FP16" : "FP32";

    std::string prefix = cv::utils::fs::join(getenv("INTEL_CVSDK_DIR"),
                         cv::utils::fs::join("deployment_tools",
                         cv::utils::fs::join("intel_models",
                         cv::utils::fs::join(modelName,
                         cv::utils::fs::join(precision, modelName)))));
    xmlPath = prefix + ".xml";
    binPath = prefix + ".bin";
  }

  virtual void run(const cv::Mat& input, cv::Mat& output, cv::TickMeter& tm) = 0;

protected:
  std::string xmlPath, binPath, device, precision;
};

// Intel's Inference Engine runner.
class IERunner : public Runner {
public:
  IERunner(const std::string& modelName, const std::string& target);

  virtual void run(const cv::Mat& input, cv::Mat& output, cv::TickMeter& tm);

private:
  InferenceEngine::CNNNetwork net;
  InferenceEngine::BlobMap inputBlobs, outputBlobs;
  InferenceEngine::InferRequest infRequest;
};

// OpenCV runner.
class OCVRunner : public Runner {
public:
  OCVRunner(const std::string& modelName, const std::string& target);

  virtual void run(const cv::Mat& input, cv::Mat& output, cv::TickMeter& tm);

private:
  cv::dnn::Net net;
};
