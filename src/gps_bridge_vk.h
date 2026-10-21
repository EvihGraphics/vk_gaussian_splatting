/*
 * Vulkan side of the Gaussian Point Splatting (CUDA) rendering path.
 *
 * - owns the gps_cuda context (loaded from the PLY file of the displayed splat set)
 * - owns one exportable VkBuffer per frame-in-flight slot, imported into CUDA as
 *   output buffers (VK_KHR_external_memory_win32 / OPAQUE_WIN32)
 * - records the resolve pass copying the CUDA output into a G-buffer color image
 *
 * Synchronization (phase 1): gps_cuda calls are synchronous (the CUDA work is complete
 * when gpsRender returns, before the frame command buffer is submitted), and the output
 * buffer is selected with the application frame-cycle index. nvapp waits for a frame slot
 * to be retired before recording into it again, so CUDA never overwrites a buffer still
 * read by Vulkan. No external semaphore is required.
 *
 * Only compiled when USE_GPS_CUDA is defined.
 */
#pragma once

#if defined(USE_GPS_CUDA)

#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan_core.h>

#include <nvslang/slang.hpp>

#include "gps_cuda_api.h"

namespace vk_gaussian_splatting {

class GpsBridgeVk
{
public:
  struct InitInfo
  {
    VkDevice               device         = VK_NULL_HANDLE;
    VkPhysicalDevice       physicalDevice = VK_NULL_HANDLE;
    nvslang::SlangCompiler* slangCompiler = nullptr;  // used once at init to build the resolve shader
    uint32_t               frameCycleSize = 3;
    VkFormat               depthFormat    = VK_FORMAT_D32_SFLOAT;  // G-buffer depth format
  };

  // Per-frame parameters, independent of the GPS C API.
  struct FrameInput
  {
    glm::mat4 viewMatrix{1.0f};      // Vulkan sample view matrix (RUB world)
    glm::mat4 instanceTransform{1.0f};  // splat set instance model matrix
    float     fovyDeg   = 45.0f;
    float     nearPlane = 0.1f;
    float     farPlane  = 1000.0f;
    glm::vec3 background{0.0f};
    uint32_t  frameIndex = 0;
  };

  // Returns true when this device can run the pipeline (CUDA device + Vulkan external memory).
  bool init(const InitInfo& info);
  void deinit();

  bool isAvailable() const { return m_available; }
  const std::string& unavailableReason() const { return m_unavailableReason; }

  // (Re)creates the CUDA scene if plyPath differs from the currently loaded one.
  // Blocking (loads the PLY file and builds the GPS hierarchy). Returns false on failure.
  // Rebuilding is also forced when the create-time parameters (prmGps) changed.
  bool ensureScene(const std::string& plyPath, VkExtent2D size);
  void releaseScene();
  bool hasScene() const { return m_ctx != nullptr; }
  const std::string& scenePath() const { return m_scenePath; }

  // Must be called when the viewport size changes (after a device wait idle).
  void resize(VkExtent2D size);

  // Runs GPS for this frame (synchronously) into the output buffer of frameSlot.
  bool render(const FrameInput& input, uint32_t frameSlot);

  // Records the resolve dispatch from the output buffer of frameSlot into dstImageView
  // (VK_IMAGE_LAYOUT_GENERAL, storage usage).
  void cmdResolve(VkCommandBuffer cmd, uint32_t frameSlot, VkImageView dstImageView);

  // Records a fullscreen pass writing the GPS front-most point depth of frameSlot into
  // depthImageView (VK_IMAGE_LAYOUT_GENERAL), so that visual helpers are occluded correctly.
  void cmdWriteDepth(VkCommandBuffer cmd, uint32_t frameSlot, VkImageView depthImageView);

  GpsStats    stats() const;
  std::string deviceName() const;
  float       lastLoadSeconds() const { return m_lastLoadSeconds; }

private:
  struct SharedBuffer
  {
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize   size   = 0;
  };

  bool createSharedBuffers(VkExtent2D size);
  void destroySharedBuffers();
  bool createResolvePipeline(nvslang::SlangCompiler* compiler);
  bool createDepthPipeline(nvslang::SlangCompiler* compiler);
  void destroyResolvePipeline();
  uint32_t findDeviceLocalMemoryType(uint32_t typeBits) const;

  VkDevice         m_device         = VK_NULL_HANDLE;
  VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
  uint32_t         m_frameCycleSize = 3;
  uint8_t          m_deviceUUID[VK_UUID_SIZE]{};

  bool        m_available = false;
  std::string m_unavailableReason;

  GpsContext*  m_ctx = nullptr;
  std::string  m_scenePath;
  VkExtent2D   m_size{0, 0};
  int32_t      m_loadedSamplesPerPixel = 0;
  float        m_lastLoadSeconds       = 0.0f;
  std::string  m_failedScenePath;  // avoids retrying a failing load every frame

  std::vector<SharedBuffer> m_buffers;

  VkDescriptorSetLayout m_dsetLayout     = VK_NULL_HANDLE;
  VkPipelineLayout      m_pipelineLayout = VK_NULL_HANDLE;
  VkPipeline            m_pipeline       = VK_NULL_HANDLE;

  VkFormat              m_depthFormat         = VK_FORMAT_D32_SFLOAT;
  VkDescriptorSetLayout m_depthDsetLayout     = VK_NULL_HANDLE;
  VkPipelineLayout      m_depthPipelineLayout = VK_NULL_HANDLE;
  VkPipeline            m_depthPipeline       = VK_NULL_HANDLE;
};

}  // namespace vk_gaussian_splatting

#endif  // USE_GPS_CUDA
