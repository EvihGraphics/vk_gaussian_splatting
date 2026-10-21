/*
 * gps_cuda - adapter between the Gaussian Point Splatting CUDA renderer and a host
 * application (vk_gaussian_splatting). See gps_cuda_api.h for the conventions.
 *
 * Only this file is written for the integration; the GPS sources are compiled
 * unmodified from the git submodule (3rdparty/gaussian-point-splatting).
 */

// Must come first, exactly like in the GPS translation units, so that glm is configured
// identically (GPS defines several GLM_FORCE_* macros in its headers).
#include "core/rendering/passes/gaussian_point_splatting.cuh"
#include "render_settings.cuh"

#include "gps_cuda_api.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <new>
#include <sstream>
#include <string>
#include <vector>

namespace {

thread_local std::string g_lastError;

GpsResult setError(GpsResult code, const std::string& msg)
{
  g_lastError = msg;
  fprintf(stderr, "[gps_cuda] %s\n", msg.c_str());
  return code;
}

GpsResult checkCuda(cudaError_t err, const char* what)
{
  if(err == cudaSuccess)
    return GPS_SUCCESS;
  return setError(err == cudaErrorMemoryAllocation ? GPS_ERROR_OUT_OF_MEMORY : GPS_ERROR_CUDA,
                  std::string(what) + ": " + cudaGetErrorString(err));
}

#define GPS_CUDA_TRY(expr)                                                                                             \
  do                                                                                                                   \
  {                                                                                                                    \
    GpsResult r_ = checkCuda((expr), #expr);                                                                           \
    if(r_ != GPS_SUCCESS)                                                                                              \
      return r_;                                                                                                       \
  } while(0)

// Reads "element vertex N" from the PLY header. GaussianPointSplatting's constructor
// ignores loader failures, so we validate the file up front.
bool readPlyVertexCount(const std::string& path, uint64_t& count, std::string& err)
{
  std::ifstream file(path, std::ios::binary);
  if(!file.is_open())
  {
    err = "cannot open '" + path + "'";
    return false;
  }
  std::string line;
  std::getline(file, line);
  if(line.rfind("ply", 0) != 0)
  {
    err = "'" + path + "' is not a PLY file";
    return false;
  }
  bool hasOpacity = false, hasScale = false, hasRot = false;
  count           = 0;
  while(std::getline(file, line))
  {
    if(!line.empty() && line.back() == '\r')
      line.pop_back();
    if(line == "end_header")
      break;
    std::istringstream ss(line);
    std::string        a, b, c;
    ss >> a >> b >> c;
    if(a == "element" && b == "vertex")
      count = std::stoull(c);
    if(a == "property")
    {
      hasOpacity |= (c == "opacity");
      hasScale |= (c == "scale_0");
      hasRot |= (c == "rot_0");
    }
  }
  if(count == 0)
  {
    err = "'" + path + "' has no vertices";
    return false;
  }
  if(!hasOpacity || !hasScale || !hasRot)
  {
    err = "'" + path + "' is not a 3DGS PLY file (missing opacity/scale/rot properties)";
    return false;
  }
  return true;
}

glm::mat4 toMat4(const float m[16])
{
  glm::mat4 r(1.0f);
  for(int c = 0; c < 4; ++c)
    for(int rr = 0; rr < 4; ++rr)
      r[c][rr] = m[c * 4 + rr];
  return r;
}

__global__ void fillFloatKernel(float* dst, float value, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
    dst[i] = value;
}

// GPS colour buffer (glm::vec3, row 0 = bottom) -> RGBA float, row 0 = top.
// GPS depth buffer (OpenGL NDC z in [-1,1] of the front-most point, row 0 = bottom)
// -> Vulkan [0,1] depth (perspectiveRH_ZO with the same near/far), row 0 = top.
__global__ void convertOutputKernel(const glm::vec3* __restrict__ srcColor,
                                    const float* __restrict__ srcDepth,
                                    float4* __restrict__ dstColor,
                                    float* __restrict__ dstDepth,
                                    int   width,
                                    int   height,
                                    float nearPlane,
                                    float farPlane)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= width || y >= height)
    return;
  const int src = y * width + x;
  const int dst = (height - 1 - y) * width + x;

  const glm::vec3 c = srcColor[src];
  dstColor[dst]     = make_float4(c.x, c.y, c.z, 1.0f);

  if(dstDepth)
  {
    // invert CudaMath::view_to_projected_depth: z_ndc = A + B / z_view (z_view < 0)
    const float A     = (farPlane + nearPlane) / (farPlane - nearPlane);
    const float B     = (2.0f * farPlane * nearPlane) / (farPlane - nearPlane);
    const float zNdc  = srcDepth[src];
    float       depth = 1.0f;
    if(zNdc < 1.0f - 1e-7f)
    {
      const float zView = B / (zNdc - A);  // negative
      // perspectiveRH_ZO: depth = f/(f-n) + f*n / ((f-n) * z_view)
      depth = farPlane / (farPlane - nearPlane) + (farPlane * nearPlane) / ((farPlane - nearPlane) * zView);
      depth = fminf(fmaxf(depth, 0.0f), 1.0f);
    }
    dstDepth[dst] = depth;
  }
}

struct OutputSlot
{
  cudaExternalMemory_t mem  = nullptr;
  void*                ptr  = nullptr;
  uint64_t             size = 0;
};

}  // namespace

struct GpsContext
{
  RenderSettings          settings{};
  Statistics              statistics{};
  GaussianPointSplatting* pass = nullptr;

  int        device = 0;
  char       deviceName[256]{};
  uint64_t   numGaussians = 0;
  int        width = 0, height = 0;
  glm::vec3* d_color      = nullptr;  // GPS output (averaged per frame), width*height vec3
  float*     d_depth      = nullptr;  // GPS NDC depth, width*height
  float4*    d_hostStaging = nullptr;  // used by gpsRenderToHost

  std::vector<OutputSlot> slots;

  cudaEvent_t evStart = nullptr, evStop = nullptr;
  float       lastGpuMs = 0.0f;
};

namespace {

void freeFrameBuffers(GpsContext* ctx)
{
  if(ctx->d_color)
    cudaFree(ctx->d_color);
  if(ctx->d_depth)
    cudaFree(ctx->d_depth);
  if(ctx->d_hostStaging)
    cudaFree(ctx->d_hostStaging);
  ctx->d_color       = nullptr;
  ctx->d_depth       = nullptr;
  ctx->d_hostStaging = nullptr;
}

// Seeds the occlusion-culling depth pyramid with "far everywhere" so that the very first
// frame (or the first frame after a resize) does not cull against uninitialized memory.
GpsResult resetDepthPyramid(GpsContext* ctx)
{
  const int n = ctx->width * ctx->height;
  fillFloatKernel<<<(n + 255) / 256, 256>>>(ctx->d_depth, 1.0f, n);
  GPS_CUDA_TRY(cudaGetLastError());
  ctx->pass->depthMipChain.build_mipchain(ctx->d_depth, ctx->settings.occlusion_culling_min_reduce_count);
  GPS_CUDA_TRY(cudaDeviceSynchronize());
  return GPS_SUCCESS;
}

GpsResult allocFrameBuffers(GpsContext* ctx, int width, int height)
{
  freeFrameBuffers(ctx);
  ctx->width                  = width;
  ctx->height                 = height;
  ctx->settings.resolution    = {width, height};
  const size_t n              = size_t(width) * size_t(height);
  GPS_CUDA_TRY(cudaMalloc((void**)&ctx->d_color, n * sizeof(glm::vec3)));
  GPS_CUDA_TRY(cudaMalloc((void**)&ctx->d_depth, n * sizeof(float)));
  GPS_CUDA_TRY(cudaMemset(ctx->d_color, 0, n * sizeof(glm::vec3)));
  return GPS_SUCCESS;
}

GpsResult selectDevice(const GpsCreateInfo* info, int& device, char* name, size_t nameSize)
{
  int count = 0;
  if(cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    return setError(GPS_ERROR_NO_DEVICE, "no CUDA device found");

  device = -1;
  for(int i = 0; i < count; ++i)
  {
    cudaDeviceProp prop{};
    if(cudaGetDeviceProperties(&prop, i) != cudaSuccess)
      continue;
    if(!info->useDeviceUUID || std::memcmp(prop.uuid.bytes, info->deviceUUID, 16) == 0)
    {
      device = i;
      snprintf(name, nameSize, "%s", prop.name);
      break;
    }
  }
  if(device < 0)
    return setError(GPS_ERROR_NO_DEVICE, "no CUDA device matches the Vulkan physical device UUID");
  GPS_CUDA_TRY(cudaSetDevice(device));
  return GPS_SUCCESS;
}

void applyFrameParams(GpsContext* ctx, const GpsFrameParams* p)
{
  RenderSettings& s = ctx->settings;

  s.fovy            = p->fovyDeg;
  s.near_view_plane = p->nearPlane;
  s.far_view_plane  = p->farPlane;
  s.backgroundColor = {p->background[0], p->background[1], p->background[2]};
  s.frame           = int(p->frameIndex);

  s.supersampling_factor        = glm::clamp(p->supersamplingFactor, 1, 4);
  s.enable_occlusion_culling    = p->enableOcclusionCulling != 0;
  s.enable_hierarchical_culling = p->enableHierarchicalCulling != 0;
  s.use_unbiased_2d_splatting   = p->unbiasedSplatting != 0;
  s.reduce_point_count          = p->reducePointCount != 0;
  s.cull_small_gaussians        = p->cullSmallGaussians != 0;
  s.freeze_culling              = false;

  const float aspect = s.get_aspectXY();

  s.sceneMatrix            = glm::mat4(1.0f);
  s.cameraViewMatrix       = toMat4(p->view);
  s.culledViewMatrix       = s.cameraViewMatrix;
  s.cameraProjectionMatrix = glm::perspective(glm::radians(s.fovy), aspect, s.near_view_plane, s.far_view_plane);
  s.cameraNormalProjectionMatrix = s.cameraProjectionMatrix;
  s.cameraPosition               = glm::vec3(glm::affineInverse(s.cameraViewMatrix)[3]);
  s.frustumPlanes = s.camera.getFrustumPlanesFromViewMatrix(s.fovy, s.cameraViewMatrix, aspect, s.near_view_plane, s.far_view_plane);
  s.copyToGPU();
}

GpsResult renderInternal(GpsContext* ctx, const GpsFrameParams* params, float4* dstColor, float* dstDepth)
{
  applyFrameParams(ctx, params);

  GPS_CUDA_TRY(cudaEventRecord(ctx->evStart, 0));

  ctx->pass->render(ctx->d_color, ctx->d_depth, ctx->settings);

  dim3 block(16, 16);
  dim3 grid((ctx->width + block.x - 1) / block.x, (ctx->height + block.y - 1) / block.y);
  convertOutputKernel<<<grid, block>>>(ctx->d_color, ctx->d_depth, dstColor, dstDepth, ctx->width, ctx->height,
                                       ctx->settings.near_view_plane, ctx->settings.far_view_plane);
  GPS_CUDA_TRY(cudaGetLastError());

  GPS_CUDA_TRY(cudaEventRecord(ctx->evStop, 0));
  // Synchronous by contract: the host submits Vulkan work reading dst right after this call.
  GPS_CUDA_TRY(cudaEventSynchronize(ctx->evStop));
  GPS_CUDA_TRY(cudaGetLastError());
  cudaEventElapsedTime(&ctx->lastGpuMs, ctx->evStart, ctx->evStop);
  return GPS_SUCCESS;
}

}  // namespace

extern "C" {

GPS_API void gpsDefaultCreateInfo(GpsCreateInfo* info)
{
  if(!info)
    return;
  std::memset(info, 0, sizeof(*info));
  info->width             = 1920;
  info->height            = 1080;
  info->samplesPerPixel   = 1;
  info->maxNumberOfPoints = 250000000;
}

GPS_API void gpsDefaultFrameParams(GpsFrameParams* p)
{
  if(!p)
    return;
  std::memset(p, 0, sizeof(*p));
  for(int i = 0; i < 4; ++i)
    p->view[i * 5] = 1.0f;
  p->fovyDeg                   = 45.0f;
  p->nearPlane                 = 0.2f;
  p->farPlane                  = 10000.0f;
  p->supersamplingFactor       = 2;
  p->enableOcclusionCulling    = 1;
  p->enableHierarchicalCulling = 1;
  p->unbiasedSplatting         = 1;
  p->reducePointCount          = 1;
}

GPS_API GpsResult gpsCreate(const GpsCreateInfo* info, GpsContext** outContext)
{
  if(!info || !outContext || !info->plyPath || info->width <= 0 || info->height <= 0)
    return setError(GPS_ERROR_INVALID_ARG, "gpsCreate: invalid arguments");
  *outContext = nullptr;

  std::string err;
  uint64_t    vertexCount = 0;
  if(!readPlyVertexCount(info->plyPath, vertexCount, err))
    return setError(GPS_ERROR_LOAD_FAILED, err);

  GpsContext* ctx = new GpsContext();
  GpsResult   res = selectDevice(info, ctx->device, ctx->deviceName, sizeof(ctx->deviceName));
  if(res != GPS_SUCCESS)
  {
    delete ctx;
    return res;
  }

  RenderSettings& s                    = ctx->settings;
  s.statistics                         = &ctx->statistics;
  s.model_path                         = RenderSettings::normalize_slashes(info->plyPath);
  s.render_pass_count                  = info->samplesPerPixel > 0 ? info->samplesPerPixel : 1;
  s.max_number_of_points               = info->maxNumberOfPoints > 0 ? info->maxNumberOfPoints : 250000000;
  s.fully_disable_hierarchical_culling = info->disableHierarchicalCulling != 0;
  s.sort_morton_order                  = info->sortMortonOrder != 0;
  s.render_gui                         = false;
  ctx->numGaussians                    = vertexCount;

  res = allocFrameBuffers(ctx, info->width, info->height);
  if(res == GPS_SUCCESS)
  {
    try
    {
      ctx->pass = new GaussianPointSplatting(s);
    }
    catch(const std::exception& e)
    {
      res = setError(GPS_ERROR_LOAD_FAILED, std::string("GaussianPointSplatting: ") + e.what());
    }
  }
  if(res == GPS_SUCCESS)
    res = checkCuda(cudaDeviceSynchronize(), "GaussianPointSplatting construction");
  if(res == GPS_SUCCESS)
    res = checkCuda(cudaEventCreate(&ctx->evStart), "cudaEventCreate");
  if(res == GPS_SUCCESS)
    res = checkCuda(cudaEventCreate(&ctx->evStop), "cudaEventCreate");
  if(res == GPS_SUCCESS)
    res = resetDepthPyramid(ctx);

  if(res != GPS_SUCCESS)
  {
    gpsDestroy(ctx);
    return res;
  }
  *outContext = ctx;
  return GPS_SUCCESS;
}

GPS_API void gpsDestroy(GpsContext* ctx)
{
  if(!ctx)
    return;
  cudaSetDevice(ctx->device);
  cudaDeviceSynchronize();
  gpsReleaseOutputBuffers(ctx);
  delete ctx->pass;
  ctx->pass = nullptr;
  freeFrameBuffers(ctx);
  if(ctx->settings.d_frustumPlanes)
    cudaFree(ctx->settings.d_frustumPlanes);
  ctx->settings.d_frustumPlanes = nullptr;
  if(ctx->evStart)
    cudaEventDestroy(ctx->evStart);
  if(ctx->evStop)
    cudaEventDestroy(ctx->evStop);
  delete ctx;
}

GPS_API GpsResult gpsResize(GpsContext* ctx, int32_t width, int32_t height)
{
  if(!ctx || width <= 0 || height <= 0)
    return setError(GPS_ERROR_INVALID_ARG, "gpsResize: invalid arguments");
  if(width == ctx->width && height == ctx->height)
    return GPS_SUCCESS;

  GPS_CUDA_TRY(cudaSetDevice(ctx->device));
  GPS_CUDA_TRY(cudaDeviceSynchronize());
  gpsReleaseOutputBuffers(ctx);

  GpsResult res = allocFrameBuffers(ctx, width, height);
  if(res != GPS_SUCCESS)
    return res;

  // DepthMipChain has its resolution fixed at construction and no resize method. It is a
  // public member of GaussianPointSplatting, so rebuild it in place. The splatting image
  // buffers are resized by GaussianPointSplatting::render() itself.
  ctx->pass->depthMipChain.~DepthMipChain();
  new(&ctx->pass->depthMipChain) DepthMipChain(width, height);
  GPS_CUDA_TRY(cudaGetLastError());

  return resetDepthPyramid(ctx);
}

GPS_API GpsResult gpsImportOutputBuffer(GpsContext* ctx, uint32_t slot, void* handle, uint64_t allocationSize, int32_t dedicated)
{
  if(!ctx || !handle || allocationSize == 0)
    return setError(GPS_ERROR_INVALID_ARG, "gpsImportOutputBuffer: invalid arguments");
  const uint64_t needed = gpsOutputBufferSize(ctx->width, ctx->height);
  if(allocationSize < needed)
    return setError(GPS_ERROR_INVALID_ARG, "gpsImportOutputBuffer: allocation too small for the current resolution");

  GPS_CUDA_TRY(cudaSetDevice(ctx->device));
  if(slot >= ctx->slots.size())
    ctx->slots.resize(slot + 1);
  OutputSlot& s = ctx->slots[slot];
  if(s.ptr)
    cudaFree(s.ptr);
  if(s.mem)
    cudaDestroyExternalMemory(s.mem);
  s = {};

  cudaExternalMemoryHandleDesc desc{};
#if defined(_WIN32)
  desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
  desc.handle.win32.handle = handle;
#else
  desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
  desc.handle.fd = int(intptr_t(handle));
#endif
  desc.size  = allocationSize;
  desc.flags = dedicated ? cudaExternalMemoryDedicated : 0;
  GPS_CUDA_TRY(cudaImportExternalMemory(&s.mem, &desc));

  cudaExternalMemoryBufferDesc bufDesc{};
  bufDesc.offset = 0;
  bufDesc.size   = allocationSize;
  bufDesc.flags  = 0;
  GPS_CUDA_TRY(cudaExternalMemoryGetMappedBuffer(&s.ptr, s.mem, &bufDesc));
  s.size = allocationSize;
  return GPS_SUCCESS;
}

GPS_API void gpsReleaseOutputBuffers(GpsContext* ctx)
{
  if(!ctx)
    return;
  cudaSetDevice(ctx->device);
  for(OutputSlot& s : ctx->slots)
  {
    if(s.ptr)
      cudaFree(s.ptr);
    if(s.mem)
      cudaDestroyExternalMemory(s.mem);
    s = {};
  }
  ctx->slots.clear();
}

GPS_API GpsResult gpsRender(GpsContext* ctx, const GpsFrameParams* params, uint32_t slot)
{
  if(!ctx || !params || slot >= ctx->slots.size() || !ctx->slots[slot].ptr)
    return setError(GPS_ERROR_INVALID_ARG, "gpsRender: invalid context, parameters or output slot");
  GPS_CUDA_TRY(cudaSetDevice(ctx->device));
  char* base = reinterpret_cast<char*>(ctx->slots[slot].ptr);
  return renderInternal(ctx, params, reinterpret_cast<float4*>(base),
                        reinterpret_cast<float*>(base + gpsOutputDepthOffset(ctx->width, ctx->height)));
}

GPS_API GpsResult gpsRenderToHost(GpsContext* ctx, const GpsFrameParams* params, float* rgbaOut, float* depthOut)
{
  if(!ctx || !params || !rgbaOut)
    return setError(GPS_ERROR_INVALID_ARG, "gpsRenderToHost: invalid arguments");
  GPS_CUDA_TRY(cudaSetDevice(ctx->device));
  const size_t colorBytes = size_t(gpsOutputDepthOffset(ctx->width, ctx->height));
  const size_t depthBytes = size_t(ctx->width) * size_t(ctx->height) * sizeof(float);
  if(!ctx->d_hostStaging)
    GPS_CUDA_TRY(cudaMalloc((void**)&ctx->d_hostStaging, size_t(gpsOutputBufferSize(ctx->width, ctx->height))));
  float* stagingDepth = reinterpret_cast<float*>(reinterpret_cast<char*>(ctx->d_hostStaging) + colorBytes);
  GpsResult res       = renderInternal(ctx, params, ctx->d_hostStaging, stagingDepth);
  if(res != GPS_SUCCESS)
    return res;
  GPS_CUDA_TRY(cudaMemcpy(rgbaOut, ctx->d_hostStaging, colorBytes, cudaMemcpyDeviceToHost));
  if(depthOut)
    GPS_CUDA_TRY(cudaMemcpy(depthOut, stagingDepth, depthBytes, cudaMemcpyDeviceToHost));
  return GPS_SUCCESS;
}

GPS_API void gpsGetStats(const GpsContext* ctx, GpsStats* stats)
{
  if(!ctx || !stats)
    return;
  stats->numGaussians = ctx->numGaussians;
  stats->pointsPass1  = uint32_t(ctx->statistics.numPointsOne);
  stats->pointsPass2  = ctx->settings.enable_occlusion_culling ? uint32_t(ctx->statistics.numPointsTwo) : 0u;
  stats->gpuTimeMs    = ctx->lastGpuMs;
}

GPS_API const char* gpsGetLastError(void)
{
  return g_lastError.c_str();
}

GPS_API const char* gpsGetDeviceName(const GpsContext* ctx)
{
  return ctx ? ctx->deviceName : "";
}

}  // extern "C"
