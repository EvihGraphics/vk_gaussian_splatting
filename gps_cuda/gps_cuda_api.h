/*
 * gps_cuda - C ABI wrapper around the CUDA implementation of
 * "Gaussian Point Splatting" (Rijsdijk et al., TOG 2026,
 * https://github.com/JorisAR/gaussian-point-splatting).
 *
 * This header is the only thing the host application includes. It deliberately
 * exposes plain C types (no glm, no CUDA headers) so that the GPS sources, which
 * ship their own glm with non-default GLM_FORCE_* settings, stay isolated inside
 * gps_cuda.dll.
 *
 * Conventions
 *  - All matrices are 4x4, column-major (glm/OpenGL memory layout).
 *  - The view matrix is an OpenGL-style view (right-handed, camera looks down -Z,
 *    +Y up) expressed in the coordinate system of the PLY file as stored on disk.
 *  - Output images are RGBA float32, tightly packed, row 0 = top of the image
 *    (Vulkan convention). Alpha is always 1.
 *  - gpsRender* calls are synchronous: when they return, all CUDA work writing the
 *    output buffer has completed.
 */
#ifndef GPS_CUDA_API_H
#define GPS_CUDA_API_H

#include <stdint.h>

#if defined(_WIN32)
#if defined(GPS_CUDA_BUILD)
#define GPS_API __declspec(dllexport)
#else
#define GPS_API __declspec(dllimport)
#endif
#else
#define GPS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GPS_CUDA_API_VERSION 1

typedef struct GpsContext GpsContext;

typedef enum GpsResult
{
  GPS_SUCCESS             = 0,
  GPS_ERROR_INVALID_ARG   = -1,
  GPS_ERROR_NO_DEVICE     = -2,
  GPS_ERROR_LOAD_FAILED   = -3,
  GPS_ERROR_CUDA          = -4,
  GPS_ERROR_OUT_OF_MEMORY = -5,
} GpsResult;

// Parameters that require rebuilding the context (reloading the PLY file) when changed.
typedef struct GpsCreateInfo
{
  const char* plyPath;  // UTF-8 / ANSI path to a 3DGS .ply file
  int32_t     width;    // output resolution (can be changed later with gpsResize)
  int32_t     height;

  // Select the CUDA device matching this UUID (VkPhysicalDeviceIDProperties::deviceUUID).
  // If useDeviceUUID == 0, device 0 is used.
  uint8_t deviceUUID[16];
  int32_t useDeviceUUID;

  int32_t samplesPerPixel;             // GPS "render passes" per pixel per frame (default 1)
  int32_t maxNumberOfPoints;           // point budget per pass (default 250e6, ~1GB VRAM)
  int32_t disableHierarchicalCulling;  // 1: do not build the Gaussian BVH at all
  int32_t sortMortonOrder;             // 1: morton-sort instead of BVH-sort at load time
} GpsCreateInfo;

// Per-frame parameters.
typedef struct GpsFrameParams
{
  float    view[16];  // see conventions above
  float    fovyDeg;   // vertical field of view in degrees
  float    nearPlane;
  float    farPlane;
  float    background[3];
  uint32_t frameIndex;  // must change every frame (random seed + internal ping-pong)

  int32_t supersamplingFactor;  // 1..4, default 2 (2 -> 2x2 samples per output pixel)
  int32_t enableOcclusionCulling;
  int32_t enableHierarchicalCulling;
  int32_t unbiasedSplatting;  // 1: unbiased (dilog-corrected) 2D splatting
  int32_t reducePointCount;   // 1: K = ss^2 points per thread, 0: one point per thread
  int32_t cullSmallGaussians;
} GpsFrameParams;

typedef struct GpsStats
{
  uint64_t numGaussians;
  uint32_t pointsPass1;
  uint32_t pointsPass2;
  float    gpuTimeMs;  // CUDA time of the last gpsRender* call (GPS passes + output conversion)
} GpsStats;

// Fills sensible defaults.
GPS_API void gpsDefaultCreateInfo(GpsCreateInfo* info);
GPS_API void gpsDefaultFrameParams(GpsFrameParams* params);

// Loads the PLY file, builds the culling hierarchy and uploads the scene. Can take seconds.
GPS_API GpsResult gpsCreate(const GpsCreateInfo* info, GpsContext** outContext);
GPS_API void      gpsDestroy(GpsContext* ctx);

// Changes the output resolution. Releases all imported output buffers.
GPS_API GpsResult gpsResize(GpsContext* ctx, int32_t width, int32_t height);

// Imports an external (Vulkan) memory allocation as output buffer #slot.
// handle: Win32 NT HANDLE on Windows, file descriptor (cast to pointer) on Linux.
// The handle is not consumed; the caller can close it after this call on Windows.
// The buffer must hold at least width*height*16 bytes at offset 0.
GPS_API GpsResult gpsImportOutputBuffer(GpsContext* ctx, uint32_t slot, void* handle, uint64_t allocationSize, int32_t dedicated);
GPS_API void      gpsReleaseOutputBuffers(GpsContext* ctx);

// Renders one frame into the imported output buffer #slot.
GPS_API GpsResult gpsRender(GpsContext* ctx, const GpsFrameParams* params, uint32_t slot);

// Renders one frame and copies the RGBA float image to host memory (width*height*4 floats).
GPS_API GpsResult gpsRenderToHost(GpsContext* ctx, const GpsFrameParams* params, float* rgbaOut);

GPS_API void        gpsGetStats(const GpsContext* ctx, GpsStats* stats);
GPS_API const char* gpsGetLastError(void);
GPS_API const char* gpsGetDeviceName(const GpsContext* ctx);

#ifdef __cplusplus
}
#endif

#endif  // GPS_CUDA_API_H
