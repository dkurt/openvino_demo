#include "runners.hpp"

#include <ie_icnn_network.hpp>
#include <ie_extension.h>

#include <opencv2/opencv.hpp>

using namespace InferenceEngine;

InferencePlugin loadPlugin(TargetDevice targetDevice) {
  static std::map<TargetDevice, InferencePlugin> sharedPlugins;

  auto pluginIt = sharedPlugins.find(targetDevice);
  if (pluginIt != sharedPlugins.end())
      return pluginIt->second;

  // Load a plugin for target device.
  InferencePlugin plugin(PluginDispatcher({""}).getSuitablePlugin(targetDevice));
  sharedPlugins[targetDevice] = plugin;

  // Load extensions.
  if (targetDevice == TargetDevice::eCPU)
  {
#ifdef _WIN32
    IExtensionPtr extension = make_so_pointer<IExtension>("cpu_extension_avx2.dll");
#else
    IExtensionPtr extension = make_so_pointer<IExtension>("libcpu_extension_avx2.so");
#endif  // _WIN32
    plugin.AddExtension(extension);
  }
  return plugin;
}

IERunner::IERunner(const std::string& modelName, int target) : Runner(modelName, target) {
  std::cout << modelName << " using Intel's Inference Engine runner" << std::endl;

  // Load a network.
  CNNNetReader reader;
  reader.ReadNetwork(xmlPath);
  reader.ReadWeights(binPath);

  // Create a network instance.
  net = reader.getNetwork();

  TargetDevice targetDevice = TargetDevice::eCPU;
  if (target == GPU_FP32 || target == GPU_FP16)
      targetDevice = TargetDevice::eGPU;
  else if (target == MYRIAD)
      targetDevice = TargetDevice::eMYRIAD;
  try
  {
    InferencePlugin plugin = loadPlugin(targetDevice);
    infRequest = plugin.LoadNetwork(net, {}).CreateInferRequest();
  }
  catch (const std::exception& ex)
  {
    CV_Error(cv::Error::StsAssert, cv::format("Failed to initialize Inference Engine backend: %s", ex.what()));
  }
}

void IERunner::run(const cv::Mat& input, cv::Mat& output, cv::TickMeter& tm) {
  // Wrap an OpenCV's Mat to Inference Engine's blob.
  auto inputIt = *net.getInputsInfo().begin();
  inputBlobs[inputIt.first] = make_shared_blob<float>(Precision::FP32,
                                                      inputIt.second->getDims(),
                                                      (float*)input.data);
  infRequest.SetInput(inputBlobs);

  auto outputIt = *net.getOutputsInfo().begin();
  std::vector<size_t> dims = outputIt.second->getDims();
  output.create(std::vector<int>(dims.begin(), dims.end()), CV_32F);
  outputBlobs[outputIt.first] = make_shared_blob<float>(Precision::FP32, dims,
                                                        (float*)output.data);
  infRequest.SetOutput(outputBlobs);

  tm.reset();
  tm.start();
  infRequest.Infer();
  tm.stop();
}
