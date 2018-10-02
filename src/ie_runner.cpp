#include "runners.hpp"

#include <ie_icnn_network.hpp>
#include <ie_extension.h>

#include <opencv2/opencv.hpp>

using namespace InferenceEngine;

IERunner::IERunner(const std::string& modelName, int target) : Runner(modelName, target) {
  std::cout << modelName << " using Intel's Inference Engine runner" << std::endl;

  // Load a network.
  CNNNetReader reader;
  reader.ReadNetwork(xmlPath);
  reader.ReadWeights(binPath);

  // Create a network instance.
  net = reader.getNetwork();

  static std::map<InferenceEngine::TargetDevice, InferenceEngine::InferenceEnginePluginPtr> sharedPlugins;

  InferenceEnginePluginPtr enginePtr;
  InferencePlugin plugin;
  ExecutableNetwork netExec;
  TargetDevice targetDevice = TargetDevice::eCPU;
  if (target == GPU_FP32 || target == GPU_FP16)
      targetDevice = TargetDevice::eGPU;
  else if (target == MYRIAD)
      targetDevice = TargetDevice::eMYRIAD;
  try
  {
    auto pluginIt = sharedPlugins.find(targetDevice);
    if (pluginIt != sharedPlugins.end())
    {
        enginePtr = pluginIt->second;
    }
    else
    {
      // Load a plugin for target device.
      enginePtr = PluginDispatcher({""}).getSuitablePlugin(targetDevice);
      sharedPlugins[targetDevice] = enginePtr;

      if (targetDevice == TargetDevice::eCPU)
      {
          std::string suffixes[] = {"_avx2", "_sse4", ""};
          bool haveFeature[] = {
              cv::checkHardwareSupport(CPU_AVX2),
              cv::checkHardwareSupport(CPU_SSE4_2),
              true
          };
          for (int i = 0; i < 3; ++i)
          {
              if (!haveFeature[i])
                  continue;
#ifdef _WIN32
              std::string libName = "cpu_extension" + suffixes[i] + ".dll";
#else
              std::string libName = "libcpu_extension" + suffixes[i] + ".so";
#endif  // _WIN32
              try
              {
                  IExtensionPtr extension = make_so_pointer<IExtension>(libName);
                  enginePtr->AddExtension(extension, 0);
                  break;
              }
              catch(...) {}
          }
          // Some of networks can work without a library of extra layers.
      }
    }
    plugin = InferencePlugin(enginePtr);

    netExec = plugin.LoadNetwork(net, {});
    infRequest = netExec.CreateInferRequest();
  }
  catch (const std::exception& ex)
  {
      CV_Error(cv::Error::StsAssert, cv::format("Failed to initialize Inference Engine backend: %s", ex.what()));
  }
}

void IERunner::run(const cv::Mat& input, cv::Mat& output, cv::TickMeter& tm) {
  for (auto& it : net.getInputsInfo())
  {
    const std::string& inputName = it.first;
    inputBlobs[inputName] = make_shared_blob<float>(Precision::FP32,
                                                    it.second->getDims(),
                                                    (float*)input.data);
  }
  infRequest.SetInput(inputBlobs);

  for (auto& it : net.getOutputsInfo())
  {
    std::vector<size_t> dims = it.second->getDims();
    std::vector<int> size(dims.rbegin(), dims.rend());
    output.create(size, CV_32F);

    const std::string& outputName = it.first;
    outputBlobs[outputName] = make_shared_blob<float>(Precision::FP32, dims,
                                                      (float*)output.data);
  }
  infRequest.SetOutput(outputBlobs);

  tm.reset();
  tm.start();
  infRequest.Infer();
  tm.stop();
}
