/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gaussian_splatting_ui.h>
#include "elem_camera_custom.hpp"
#include "hardware_support.h"

#if USE_DLSS
#include "dlss_wrapper.hpp"
#endif

using namespace vk_gaussian_splatting;

// create, setup and run an nvapp::Application
// with a GaussianSplatting element.
int main(int argc, char** argv)
{
  nvutils::Logger::getInstance().breakOnError(false);
  //nvutils::Logger::getInstance().setLogLevel(nvutils::Logger::LogLevel::eDEBUG);

  nvutils::ProfilerManager              profilerManager;
  nvutils::ParameterRegistry            parameterRegistry;
  nvutils::ParameterParser              parameterParser(nvutils::getExecutablePath().stem().string(), {".txt"});
  nvutils::ParameterSequencer::InitInfo sequencerInfo{// sequencer always requires a parser and registry
                                                      .parameterParser   = &parameterParser,
                                                      .parameterRegistry = &parameterRegistry,
                                                      // sequencer uses the profiler for benchmarking
                                                      .profilerManager = &profilerManager};

  nvvk::Context                vkContext;  // The Vulkan context
  nvvk::ContextInitInfo        vkSetup;    // Information to create the Vulkan context
  nvapp::Application           application;
  nvapp::ApplicationCreateInfo appInfo;  // Information to create the application
  bool                         benchmarkMode = false;

  /////////////////////////////////
  // Parse the command line to get the application creation information
  // those parameter will have no effect if changed via benchmark script
  // see GaussianSplatting constructor for other options
  parameterRegistry.addVector({"size", "Size of the window to be created"}, &appInfo.windowSize);
  parameterRegistry.add({"vsync"}, &appInfo.vSync);
  parameterRegistry.add({"headless", "Run without a window (no swapchain, no display)"}, &appInfo.headless);
  parameterRegistry.add({"headlessFrameCount", "Number of frames to render in headless mode (ignored when --benchmark is enabled)"},
                        &appInfo.headlessFrameCount);
  parameterRegistry.add({"verbose", "Verbose output of the Vulkan context"}, &vkSetup.verbose);
  parameterRegistry.add({"validation", "Enable validation layers"}, &vkSetup.enableValidationLayers);
  parameterRegistry.add({"benchmark", "Enable benchmarking, prevents async loadings and turns off vsync"}, &benchmarkMode);
  parameterRegistry.add({"forcegpu", "Force the use of a specific GPU by probviding its ID"}, &vkSetup.forceGPU);

  registerCommandLineParameters(&parameterRegistry);

  /////////////////////////////////
  // Create elements of the application, including the core of the sample (gaussianSplatting)

  // The GaussianSplattingUI includes the core GaussianSplatting class by inheritance
  auto gaussianSplatting = std::make_shared<GaussianSplattingUI>(&profilerManager, &parameterRegistry, &benchmarkMode);

  // add a few more parameters to registry and parser to handle sequencer settings
  sequencerInfo.registerScriptParameters(parameterRegistry, parameterParser);

  // extends reporting output with memory consumption information
  sequencerInfo.postCallbacks.emplace_back(
      [&](const nvutils::ParameterSequencer::State& /* unused */) { gaussianSplatting->benchmarkAdvance(); });

  // After the creation of the elements we have more parameters in the registry than before (from gaussianSplatting).
  // Therefore add the entire registry to the commandline parser again, to add new ones.
  parameterParser.add(parameterRegistry);
  // commandline parsing
  parameterParser.parse(argc, argv);
  // backup the default applications parameters, including those modified by command line
  storeDefaultParameters();
  // set more verbose for benchmark usage later on
  parameterParser.setVerbose(true);

  // this element requires sequencerInfo that is potentially updated by parameterParser
  auto elemSequencer = std::make_shared<nvapp::ElementSequencer>(sequencerInfo);

  /////////////////////////////////
  // Vulkan creation context information
  vkSetup.enableAllFeatures = true;

  // - Instance extensions
  vkSetup.instanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  // - Device extensions
  static VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR baryFeaturesKHR = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
  static VkPhysicalDeviceMeshShaderFeaturesEXT meshFeaturesEXT = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
  };

  static VkPhysicalDeviceFragmentShadingRateFeaturesKHR fragFeaturesKHR = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR,
  };
  vkSetup.deviceExtensions.emplace_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);  // for vk_radix_sort (vrdx)
  vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_EXT_MESH_SHADER_EXTENSION_NAME, &meshFeaturesEXT, false);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, &fragFeaturesKHR, true);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, &baryFeaturesKHR, false);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);  // for ImGui

  // Dynamic depth/stencil state (vkCmdSetDepthTestEnable, vkCmdSetDepthCompareOp, etc.) is core and
  // mandatory since Vulkan 1.3 (promoted from VK_EXT_extended_dynamic_state / _2), so no extension is
  // requested here; the 1.4 baseline guarantees it.

  // Ray tracing extensions: all three are requested as optional. The
  // composite isSupported.raytracing flag (hardware_support.cpp) is true only
  // when the trio is actually enabled on the device, and every RTX code path
  // (shader compilation, pipeline creation, AS builds, UI entries) is gated
  // by it.
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  vkSetup.deviceExtensions.emplace_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature, false);  // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  vkSetup.deviceExtensions.emplace_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature, false);  // To use vkCmdTraceRaysKHR
  vkSetup.deviceExtensions.emplace_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, nullptr, false);  // Required by ray tracing pipeline
  VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR rtPositionFetchFeature = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR, .rayTracingPositionFetch = VK_TRUE};
  vkSetup.deviceExtensions.emplace_back(VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, &rtPositionFetchFeature, false);

  VkPhysicalDeviceShaderClockFeaturesKHR clockFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  vkSetup.deviceExtensions.emplace_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME, &clockFeatures);

  VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV serFeatures = {
      .sType                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV,
      .rayTracingInvocationReorder = VK_TRUE,
  };
  vkSetup.deviceExtensions.emplace_back(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME, &serFeatures, false);

  // Sphere primitives for ray tracing acceleration structures (optional NV extension)
  static VkPhysicalDeviceRayTracingLinearSweptSpheresFeaturesNV lssFeaturesNV = {
      .sType   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_LINEAR_SWEPT_SPHERES_FEATURES_NV,
      .spheres = VK_TRUE,
  };
  vkSetup.deviceExtensions.emplace_back(VK_NV_RAY_TRACING_LINEAR_SWEPT_SPHERES_EXTENSION_NAME, &lssFeaturesNV, false);

  // Memory budget extension for querying VRAM usage
  vkSetup.deviceExtensions.emplace_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

#if defined(USE_GPS_CUDA)
  // Gaussian Point Splatting (CUDA) pipeline: share output buffers with CUDA (optional)
#if defined(_WIN32)
  vkSetup.deviceExtensions.emplace_back("VK_KHR_external_memory_win32", nullptr, false);
#else
  vkSetup.deviceExtensions.emplace_back("VK_KHR_external_memory_fd", nullptr, false);
#endif
#endif

  // Fragment shader interlock for order-independent transparency (FTB rendering)
  static VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT interlockFeatures = {
      .sType                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT,
      .fragmentShaderPixelInterlock = VK_TRUE,
  };
  vkSetup.deviceExtensions.emplace_back(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME, &interlockFeatures, true);

  // Note: smooth line rasterization is a Vulkan 1.4 core feature (enabled via enableAllFeatures).

  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

#if USE_DLSS
  // Adding the DLSS extensions to the instance
  static std::vector<VkExtensionProperties> extraInstanceExtensions;
  DlssRayReconstruction::getRequiredInstanceExtensions({}, extraInstanceExtensions);
  for(auto& ext : extraInstanceExtensions)
  {
    vkSetup.instanceExtensions.emplace_back(ext.extensionName);
  }

  // Adding the extra device extensions required by DLSS (Using callback)
  static std::vector<VkExtensionProperties> extraDeviceExtensions;
  vkSetup.postSelectPhysicalDeviceCallback = [](VkInstance instance, VkPhysicalDevice physicalDevice, nvvk::ContextInitInfo& vkSetup) {
    DlssRayReconstruction::getRequiredDeviceExtensions({}, instance, physicalDevice, extraDeviceExtensions);
    for(auto& ext : extraDeviceExtensions)
    {
      vkSetup.deviceExtensions.push_back({.extensionName = ext.extensionName, .specVersion = ext.specVersion});
    }
    return true;
  };
#endif

  // Setting up the validation layers
  nvvk::ValidationSettings vvlInfo{};
  vvlInfo.setPreset(nvvk::ValidationSettings::LayerPresets::eStandard);
  //vvlInfo.setPreset(nvvk::ValidationSettings::LayerPresets::eSynchronization);
  // Disable expensive shader validation during pipeline creation
  vvlInfo.check_shaders         = VK_FALSE;
  vvlInfo.check_shaders_caching = VK_FALSE;

  // Adding the validation layer settings
  vkSetup.instanceCreateInfoExt = vvlInfo.buildPNextChain();

  // Create Vulkan context. Vulkan 1.4 is required: maintenance5 (buffer usage flags2, inline SPIR-V
  // pipeline modules), vertex attribute divisor and smooth line rasterization are used as core 1.4
  // features.
  vkSetup.apiVersion = VK_API_VERSION_1_4;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation (Vulkan 1.4 is required).\n");
    return 1;
  }

  // Populate the global hardware capability registry from the freshly created
  // context. After this point, code can branch on isSupported.* (see
  // src/hardware_support.h). DLSS runtime availability is filled in later
  // from GaussianSplatting once NGX has probed the device.
  isSupported.initFromVulkanContext(vkContext);

  /////////////////////////////////
  // Application setup
  appInfo.name                  = TARGET_NAME;
  appInfo.instance              = vkContext.getInstance();
  appInfo.device                = vkContext.getDevice();
  appInfo.physicalDevice        = vkContext.getPhysicalDevice();
  appInfo.queues                = vkContext.getQueueInfos();
  appInfo.hasUndockableViewport = true;
  appInfo.useMenu               = !benchmarkMode;  // we hide the menu in benchmark mode
  if(appInfo.headless && benchmarkMode)
  {
    appInfo.headlessFrameCount = UINT32_MAX;  // Let the sequencer's close() terminate the loop
  }

  // Setting up the layout of the application
  appInfo.dockSetup = [](ImGuiID viewportID) {
    // right side panel container
    ImGuiID assetsID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.20F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Assets", assetsID);
    ImGuiID propertiesID = ImGui::DockBuilderSplitNode(assetsID, ImGuiDir_Down, 0.70F, nullptr, &assetsID);
    ImGui::DockBuilderDockWindow("Properties", propertiesID);

    // left side panel container
    ImGuiID profilerID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 0.20F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Profiler", profilerID);
    ImGuiID renderingID = ImGui::DockBuilderSplitNode(profilerID, ImGuiDir_Down, 0.65F, nullptr, &profilerID);
    ImGui::DockBuilderDockWindow("Rendering Statistics", renderingID);
    ImGui::DockBuilderDockWindow("Shader Feedback", renderingID);
    ImGuiID memoryID = ImGui::DockBuilderSplitNode(renderingID, ImGuiDir_Down, 0.30F, nullptr, &renderingID);
    ImGui::DockBuilderDockWindow("Memory Statistics", memoryID);
  };

  //
  gaussianSplatting->guiRegisterIniFileHandlers();

  // Initializes the application
  application.init(appInfo);

  // Add all application elements including our sample specific gaussianSplatting
  // onAttach will be invoked on elements at this stage
  application.addElement(elemSequencer);
  application.addElement(gaussianSplatting);

  auto elemCamera = std::make_shared<vk_gaussian_splatting::ElementCameraCustom>();
  elemCamera->setCameraManipulator(gaussianSplatting->cameraManip);
  // Disable camera updates when dragging comparison slider or transform gizmo
  elemCamera->setDisableCallback([gaussianSplatting]() {
    return gaussianSplatting->isDraggingComparisonSlider() || gaussianSplatting->isDraggingTransformHelper()
           || gaussianSplatting->isDraggingCursorTarget();
  });
  application.addElement(elemCamera);

  if(benchmarkMode)
  {
    // In this mode we do not display the GUI elements
    application.setVsync(false);
  }

  if(appInfo.headless)
  {
    ImGui::GetIO().IniFilename = nullptr;
  }

  //
  application.run();

  // Cleanup
  application.deinit();
  vkContext.deinit();

  return 0;
}
