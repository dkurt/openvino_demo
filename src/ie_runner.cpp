#include "runners.hpp"

#include <ie_icnn_network.hpp>
#include <ie_extension.h>

#include <opencv2/opencv.hpp>

using namespace InferenceEngine;

InferencePlugin loadPlugin(const std::string& device) {
  static std::map<std::string, InferencePlugin> sharedPlugins;

  auto pluginIt = sharedPlugins.find(device);
  if (pluginIt != sharedPlugins.end())
      return pluginIt->second;

  // Load a plugin for target device.
  InferencePlugin plugin(PluginDispatcher({""}).getPluginByDevice(device));
  sharedPlugins[device] = plugin;

  // Load extensions.
  if (device == "CPU")
  {
    IExtensionPtr extension = make_so_pointer<IExtension>(
#ifdef _WIN32
    "cpu_extension_avx2.dll"
#else
    "libcpu_extension_avx2.so"
#endif  // _WIN32
    );
    plugin.AddExtension(extension);
  }
  return plugin;
}

IERunner::IERunner(const std::string& modelName, const std::string& target) : Runner(modelName, target) {
  std::cout << modelName << " using Intel's Inference Engine runner" << std::endl;

  // Load a network.
  CNNNetReader reader;
  reader.ReadNetwork(xmlPath);
  reader.ReadWeights(binPath);

  // Create a network instance.
  net = reader.getNetwork();

  try
  {
    InferencePlugin plugin = loadPlugin(device);
    infRequest = plugin.LoadNetwork(net, {}).CreateInferRequest();
  }
  catch (const std::exception& ex)
  {
    CV_Error(cv::Error::StsAssert, cv::format("Failed to initialize Inference Engine backend: %s", ex.what()));
  }
}

static Blob::Ptr wrapMatToBlob(const cv::Mat& m) {
  std::vector<size_t> reversedShape(&m.size[0], &m.size[0] + m.dims);
  std::reverse(reversedShape.begin(), reversedShape.end());
  return make_shared_blob<float>(Precision::FP32, reversedShape, (float*)m.data);
}

void IERunner::run(const cv::Mat& input, cv::Mat& output, cv::TickMeter& tm) {
  auto inputIt = *net.getInputsInfo().begin();
  inputBlobs[inputIt.first] = wrapMatToBlob(input);
  infRequest.SetInput(inputBlobs);

  auto outputIt = *net.getOutputsInfo().begin();
  std::vector<size_t> dims = outputIt.second->getDims();
  output.create(std::vector<int>(dims.begin(), dims.end()), CV_32F);
  outputBlobs[outputIt.first] = wrapMatToBlob(output);
  infRequest.SetOutput(outputBlobs);

  tm.reset();
  tm.start();
  infRequest.Infer();
  tm.stop();
}
