#if defined(USE_GPS_CUDA)

#include "gps_bridge_vk.h"
#include "parameters.h"

#include <chrono>
#include <cstring>

#include <volk.h>
#include <nvutils/logger.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace vk_gaussian_splatting {

namespace {

// The Vulkan sample converts PLY data from the INRIA/COLMAP convention (RDF: x right,
// y down, z forward) to RUB when loading (see ply_loader_async.cpp). gps_cuda loads the
// PLY file as stored on disk, so its "world" is RDF. This maps RDF -> RUB (rotation of
// 180 degrees around X, i.e. what SplatSet::convertCoordinates applies to positions).
const glm::mat4 kRdfToRub = glm::mat4(glm::vec4(1, 0, 0, 0), glm::vec4(0, -1, 0, 0), glm::vec4(0, 0, -1, 0), glm::vec4(0, 0, 0, 1));

#if defined(_WIN32)
constexpr VkExternalMemoryHandleTypeFlagBits kHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
constexpr VkExternalMemoryHandleTypeFlagBits kHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

}  // namespace

bool GpsBridgeVk::init(const InitInfo& info)
{
  m_device         = info.device;
  m_physicalDevice = info.physicalDevice;
  m_frameCycleSize = std::max(1u, info.frameCycleSize);
  m_available      = false;

  // Device UUID, used by gps_cuda to select the matching CUDA device
  VkPhysicalDeviceIDProperties idProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
  VkPhysicalDeviceProperties2  props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &idProps};
  vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2);
  std::memcpy(m_deviceUUID, idProps.deviceUUID, VK_UUID_SIZE);

#if defined(_WIN32)
  if(vkGetMemoryWin32HandleKHR == nullptr)
  {
    m_unavailableReason = "VK_KHR_external_memory_win32 not enabled on this device";
    LOGW("GPS (CUDA) pipeline unavailable: %s\n", m_unavailableReason.c_str());
    return false;
  }
#else
  if(vkGetMemoryFdKHR == nullptr)
  {
    m_unavailableReason = "VK_KHR_external_memory_fd not enabled on this device";
    LOGW("GPS (CUDA) pipeline unavailable: %s\n", m_unavailableReason.c_str());
    return false;
  }
#endif

  if(props2.properties.vendorID != 0x10DE)
  {
    m_unavailableReason = "requires an NVIDIA GPU (CUDA)";
    LOGW("GPS (CUDA) pipeline unavailable: %s\n", m_unavailableReason.c_str());
    return false;
  }

  if(!createResolvePipeline(info.slangCompiler))
  {
    m_unavailableReason = "failed to build the resolve compute shader";
    LOGW("GPS (CUDA) pipeline unavailable: %s\n", m_unavailableReason.c_str());
    return false;
  }

  m_available = true;
  return true;
}

void GpsBridgeVk::deinit()
{
  releaseScene();
  destroyResolvePipeline();
  m_available = false;
}

void GpsBridgeVk::releaseScene()
{
  destroySharedBuffers();
  if(m_ctx)
  {
    gpsDestroy(m_ctx);
    m_ctx = nullptr;
  }
  m_scenePath.clear();
}

bool GpsBridgeVk::ensureScene(const std::string& plyPath, VkExtent2D size)
{
  if(!m_available || plyPath.empty() || size.width == 0 || size.height == 0)
    return false;

  const bool paramsChanged = (m_loadedSamplesPerPixel != prmGps.samplesPerPixel);
  if(m_ctx && m_scenePath == plyPath && !paramsChanged)
    return true;
  if(!m_ctx && plyPath == m_failedScenePath && !paramsChanged)
    return false;

  vkDeviceWaitIdle(m_device);
  releaseScene();

  GpsCreateInfo ci;
  gpsDefaultCreateInfo(&ci);
  ci.plyPath         = plyPath.c_str();
  ci.width           = int32_t(size.width);
  ci.height          = int32_t(size.height);
  ci.useDeviceUUID   = 1;
  ci.samplesPerPixel = prmGps.samplesPerPixel;
  std::memcpy(ci.deviceUUID, m_deviceUUID, 16);

  LOGI("GPS (CUDA): loading %s ...\n", plyPath.c_str());
  auto      t0  = std::chrono::high_resolution_clock::now();
  GpsResult res = gpsCreate(&ci, &m_ctx);
  auto      t1  = std::chrono::high_resolution_clock::now();
  m_loadedSamplesPerPixel = prmGps.samplesPerPixel;
  if(res != GPS_SUCCESS)
  {
    LOGE("GPS (CUDA): failed to load %s: %s\n", plyPath.c_str(), gpsGetLastError());
    m_ctx             = nullptr;
    m_failedScenePath = plyPath;
    return false;
  }
  m_lastLoadSeconds = std::chrono::duration<float>(t1 - t0).count();
  m_failedScenePath.clear();
  m_scenePath = plyPath;
  m_size      = size;
  LOGI("GPS (CUDA): loaded on %s in %.2f s\n", gpsGetDeviceName(m_ctx), m_lastLoadSeconds);

  if(!createSharedBuffers(size))
  {
    LOGE("GPS (CUDA): failed to create/import the shared output buffers\n");
    releaseScene();
    m_failedScenePath = plyPath;
    return false;
  }
  return true;
}

void GpsBridgeVk::resize(VkExtent2D size)
{
  if(!m_ctx || (size.width == m_size.width && size.height == m_size.height) || size.width == 0 || size.height == 0)
    return;
  vkDeviceWaitIdle(m_device);
  destroySharedBuffers();  // releases the CUDA imports too
  if(gpsResize(m_ctx, int32_t(size.width), int32_t(size.height)) != GPS_SUCCESS)
  {
    LOGE("GPS (CUDA): resize failed: %s\n", gpsGetLastError());
    releaseScene();
    return;
  }
  m_size = size;
  if(!createSharedBuffers(size))
  {
    LOGE("GPS (CUDA): failed to recreate the shared output buffers\n");
    releaseScene();
  }
}

bool GpsBridgeVk::render(const FrameInput& in, uint32_t frameSlot)
{
  if(!m_ctx || m_buffers.empty())
    return false;
  frameSlot %= uint32_t(m_buffers.size());

  GpsFrameParams fp;
  gpsDefaultFrameParams(&fp);

  // PLY (RDF) space -> sample world (RUB) -> instance -> view (OpenGL style, -Z forward)
  const glm::mat4 view = in.viewMatrix * in.instanceTransform * kRdfToRub;
  std::memcpy(fp.view, &view[0][0], sizeof(fp.view));
  fp.fovyDeg       = in.fovyDeg;
  fp.nearPlane     = in.nearPlane;
  fp.farPlane      = in.farPlane;
  fp.background[0] = in.background.x;
  fp.background[1] = in.background.y;
  fp.background[2] = in.background.z;
  fp.frameIndex    = in.frameIndex;

  fp.supersamplingFactor       = prmGps.supersamplingFactor;
  fp.enableOcclusionCulling    = prmGps.occlusionCulling ? 1 : 0;
  fp.enableHierarchicalCulling = prmGps.hierarchicalCulling ? 1 : 0;
  fp.unbiasedSplatting         = prmGps.unbiasedSplatting ? 1 : 0;
  fp.reducePointCount          = prmGps.reducePointCount ? 1 : 0;
  fp.cullSmallGaussians        = prmGps.cullSmallGaussians ? 1 : 0;

  if(gpsRender(m_ctx, &fp, frameSlot) != GPS_SUCCESS)
  {
    LOGE("GPS (CUDA): render failed: %s\n", gpsGetLastError());
    return false;
  }
  return true;
}

void GpsBridgeVk::cmdResolve(VkCommandBuffer cmd, uint32_t frameSlot, VkImageView dstImageView)
{
  if(m_buffers.empty() || m_pipeline == VK_NULL_HANDLE)
    return;
  frameSlot %= uint32_t(m_buffers.size());

  VkDescriptorBufferInfo srcInfo{m_buffers[frameSlot].buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorImageInfo  dstInfo{VK_NULL_HANDLE, dstImageView, VK_IMAGE_LAYOUT_GENERAL};

  VkWriteDescriptorSet writes[2]{};
  writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstBinding      = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo     = &srcInfo;
  writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[1].dstBinding      = 1;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  writes[1].pImageInfo      = &dstInfo;

  const glm::uvec2 pc{m_size.width, m_size.height};

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 2, writes);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  vkCmdDispatch(cmd, (m_size.width + 15) / 16, (m_size.height + 15) / 16, 1);
}

GpsStats GpsBridgeVk::stats() const
{
  GpsStats s{};
  if(m_ctx)
    gpsGetStats(m_ctx, &s);
  return s;
}

std::string GpsBridgeVk::deviceName() const
{
  return m_ctx ? std::string(gpsGetDeviceName(m_ctx)) : std::string();
}

uint32_t GpsBridgeVk::findDeviceLocalMemoryType(uint32_t typeBits) const
{
  VkPhysicalDeviceMemoryProperties memProps{};
  vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProps);
  for(uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
  {
    if((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
      return i;
  }
  return UINT32_MAX;
}

bool GpsBridgeVk::createSharedBuffers(VkExtent2D size)
{
  destroySharedBuffers();
  const VkDeviceSize bytes = VkDeviceSize(size.width) * size.height * 4 * sizeof(float);

  m_buffers.resize(m_frameCycleSize);
  for(uint32_t i = 0; i < m_frameCycleSize; ++i)
  {
    SharedBuffer& b = m_buffers[i];

    VkExternalMemoryBufferCreateInfo extInfo{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    extInfo.handleTypes = kHandleType;
    VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &extInfo};
    bufInfo.size        = bytes;
    bufInfo.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if(vkCreateBuffer(m_device, &bufInfo, nullptr, &b.buffer) != VK_SUCCESS)
      return false;
    NVVK_DBG_NAME(b.buffer);

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(m_device, b.buffer, &req);
    const uint32_t memType = findDeviceLocalMemoryType(req.memoryTypeBits);
    if(memType == UINT32_MAX)
      return false;

    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer = b.buffer;
    VkExportMemoryAllocateInfo exportInfo{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, &dedicated};
    exportInfo.handleTypes = kHandleType;
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &exportInfo};
    allocInfo.allocationSize  = req.size;
    allocInfo.memoryTypeIndex = memType;
    if(vkAllocateMemory(m_device, &allocInfo, nullptr, &b.memory) != VK_SUCCESS)
      return false;
    if(vkBindBufferMemory(m_device, b.buffer, b.memory, 0) != VK_SUCCESS)
      return false;
    b.size = req.size;

    // Export and import into CUDA
#if defined(_WIN32)
    VkMemoryGetWin32HandleInfoKHR handleInfo{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    handleInfo.memory     = b.memory;
    handleInfo.handleType = kHandleType;
    HANDLE handle         = nullptr;
    if(vkGetMemoryWin32HandleKHR(m_device, &handleInfo, &handle) != VK_SUCCESS || !handle)
      return false;
    GpsResult res = gpsImportOutputBuffer(m_ctx, i, handle, b.size, 1);
    CloseHandle(handle);  // CUDA holds its own reference after the import
#else
    VkMemoryGetFdInfoKHR fdInfo{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    fdInfo.memory     = b.memory;
    fdInfo.handleType = kHandleType;
    int fd            = -1;
    if(vkGetMemoryFdKHR(m_device, &fdInfo, &fd) != VK_SUCCESS)
      return false;
    GpsResult res = gpsImportOutputBuffer(m_ctx, i, (void*)(intptr_t)fd, b.size, 1);  // CUDA owns the fd
#endif
    if(res != GPS_SUCCESS)
    {
      LOGE("GPS (CUDA): import of output buffer %u failed: %s\n", i, gpsGetLastError());
      return false;
    }
  }
  return true;
}

void GpsBridgeVk::destroySharedBuffers()
{
  if(m_ctx)
    gpsReleaseOutputBuffers(m_ctx);
  for(SharedBuffer& b : m_buffers)
  {
    if(b.buffer)
      vkDestroyBuffer(m_device, b.buffer, nullptr);
    if(b.memory)
      vkFreeMemory(m_device, b.memory, nullptr);
  }
  m_buffers.clear();
}

bool GpsBridgeVk::createResolvePipeline(nvslang::SlangCompiler* compiler)
{
  if(!compiler || !compiler->compileFile("gps_resolve.comp.slang") || compiler->getSpirvSize() == 0)
    return false;

  VkShaderModuleCreateInfo smInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smInfo.codeSize = compiler->getSpirvSize();
  smInfo.pCode    = compiler->getSpirv();
  VkShaderModule module{};
  if(vkCreateShaderModule(m_device, &smInfo, nullptr, &module) != VK_SUCCESS)
    return false;

  VkDescriptorSetLayoutBinding bindings[2]{};
  bindings[0].binding         = 0;
  bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
  bindings[1].binding         = 1;
  bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo dslInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
  dslInfo.bindingCount = 2;
  dslInfo.pBindings    = bindings;
  NVVK_CHECK(vkCreateDescriptorSetLayout(m_device, &dslInfo, nullptr, &m_dsetLayout));

  VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(glm::uvec2)};
  VkPipelineLayoutCreateInfo plInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plInfo.setLayoutCount         = 1;
  plInfo.pSetLayouts            = &m_dsetLayout;
  plInfo.pushConstantRangeCount = 1;
  plInfo.pPushConstantRanges    = &pcRange;
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &plInfo, nullptr, &m_pipelineLayout));

  VkComputePipelineCreateInfo cpInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  cpInfo.stage.module = module;
  cpInfo.stage.pName  = "main";
  cpInfo.layout       = m_pipelineLayout;
  VkResult res        = vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &m_pipeline);
  vkDestroyShaderModule(m_device, module, nullptr);
  NVVK_DBG_NAME(m_pipeline);
  return res == VK_SUCCESS;
}

void GpsBridgeVk::destroyResolvePipeline()
{
  if(m_pipeline)
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
  if(m_pipelineLayout)
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  if(m_dsetLayout)
    vkDestroyDescriptorSetLayout(m_device, m_dsetLayout, nullptr);
  m_pipeline       = VK_NULL_HANDLE;
  m_pipelineLayout = VK_NULL_HANDLE;
  m_dsetLayout     = VK_NULL_HANDLE;
}

}  // namespace vk_gaussian_splatting

#endif  // USE_GPS_CUDA
