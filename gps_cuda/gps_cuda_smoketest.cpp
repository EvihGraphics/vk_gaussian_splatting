// Standalone smoke test for gps_cuda.dll (no Vulkan involved).
//
//   gps_cuda_smoketest <model.ply> <cameras.json> <out.png> [camIndex=0] [width=1280] [frames=64] [ss=2]
//
// Renders `frames` stochastic frames from an INRIA 3DGS camera, averages them on the host
// (the progressive accumulation normally done by the host renderer) and writes a PNG.

#include "gps_cuda_api.h"

#include <nlohmann/json.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// Builds an OpenGL-style view matrix (column-major) from an INRIA cameras.json entry.
// "rotation" holds the rows of the camera-to-world rotation (COLMAP camera frame:
// x right, y down, z forward) and "position" the camera centre, both in PLY space.
static void inriaCameraToGlView(const nlohmann::json& cam, float view[16])
{
  double R[3][3], C[3];
  for(int r = 0; r < 3; ++r)
  {
    for(int c = 0; c < 3; ++c)
      R[r][c] = cam["rotation"][r][c].get<double>();
    C[r] = cam["position"][r].get<double>();
  }
  // world-to-camera (COLMAP): Rt = R^T, t = -R^T C ; then flip y,z to get the GL camera frame.
  const double flip[3] = {1.0, -1.0, -1.0};
  double       W[4][4] = {};  // W[row][col]
  for(int r = 0; r < 3; ++r)
  {
    double t = 0.0;
    for(int c = 0; c < 3; ++c)
    {
      W[r][c] = flip[r] * R[c][r];
      t -= R[c][r] * C[c];
    }
    W[r][3] = flip[r] * t;
  }
  W[3][3] = 1.0;
  for(int c = 0; c < 4; ++c)
    for(int r = 0; r < 4; ++r)
      view[c * 4 + r] = float(W[r][c]);
}

int main(int argc, char** argv)
{
  // The DLL logs through its own std::cout; unbuffered stdout keeps both streams ordered.
  setvbuf(stdout, nullptr, _IONBF, 0);
  if(argc < 4)
  {
    printf("usage: %s <model.ply> <cameras.json> <out.png> [camIndex=0] [width=1280] [frames=64] [ss=2]\n", argv[0]);
    return 1;
  }
  const std::string plyPath  = argv[1];
  const std::string camPath  = argv[2];
  const std::string outPath  = argv[3];
  const int         camIndex = argc > 4 ? atoi(argv[4]) : 0;
  const int         width    = argc > 5 ? atoi(argv[5]) : 1280;
  const int         frames   = argc > 6 ? atoi(argv[6]) : 64;
  const int         ss       = argc > 7 ? atoi(argv[7]) : 2;

  nlohmann::json cams;
  {
    std::ifstream f(camPath);
    if(!f)
    {
      printf("cannot open %s\n", camPath.c_str());
      return 1;
    }
    f >> cams;
  }
  const nlohmann::json& cam    = cams.at(std::min<size_t>(camIndex, cams.size() - 1));
  const int             camW   = cam["width"].get<int>();
  const int             camH   = cam["height"].get<int>();
  const double          fy     = cam["fy"].get<double>();
  const int             height = int(std::lround(double(width) * camH / camW));
  const float           fovy   = float(2.0 * std::atan(0.5 * camH / fy) * 180.0 / 3.14159265358979);

  GpsCreateInfo ci;
  gpsDefaultCreateInfo(&ci);
  ci.plyPath = plyPath.c_str();
  ci.width   = width;
  ci.height  = height;

  auto        t0  = std::chrono::high_resolution_clock::now();
  GpsContext* ctx = nullptr;
  if(gpsCreate(&ci, &ctx) != GPS_SUCCESS)
  {
    printf("gpsCreate failed: %s\n", gpsGetLastError());
    return 2;
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  printf("Loaded on %s in %.2f s, %dx%d, fovy %.2f deg\n", gpsGetDeviceName(ctx),
         std::chrono::duration<double>(t1 - t0).count(), width, height, fovy);

  GpsFrameParams fp;
  gpsDefaultFrameParams(&fp);
  inriaCameraToGlView(cam, fp.view);
  fp.fovyDeg             = fovy;
  fp.nearPlane           = 0.01f;
  fp.farPlane            = 1000.0f;
  fp.supersamplingFactor = ss;

  std::vector<float>  frame(size_t(width) * height * 4);
  std::vector<double> accum(size_t(width) * height * 3, 0.0);
  double              gpuSum = 0.0;
  for(int i = 0; i < frames; ++i)
  {
    fp.frameIndex = uint32_t(i);
    if(gpsRenderToHost(ctx, &fp, frame.data()) != GPS_SUCCESS)
    {
      printf("gpsRenderToHost failed: %s\n", gpsGetLastError());
      gpsDestroy(ctx);
      return 3;
    }
    GpsStats st;
    gpsGetStats(ctx, &st);
    if(i > 0)  // first frame includes warm-up
      gpuSum += st.gpuTimeMs;
    if(i == 0 || i == frames - 1)
      printf("frame %d: %.3f ms, points pass1 %u pass2 %u, gaussians %llu\n", i, st.gpuTimeMs, st.pointsPass1,
             st.pointsPass2, (unsigned long long)st.numGaussians);
    for(size_t p = 0; p < size_t(width) * height; ++p)
      for(int c = 0; c < 3; ++c)
        accum[p * 3 + c] += frame[p * 4 + c];
  }
  if(frames > 1)
    printf("average GPS frame time (excluding first): %.3f ms\n", gpuSum / (frames - 1));

  std::vector<unsigned char> png(size_t(width) * height * 3);
  for(size_t i = 0; i < png.size(); ++i)
    png[i] = (unsigned char)std::lround(std::clamp(accum[i] / frames, 0.0, 1.0) * 255.0);
  stbi_write_png(outPath.c_str(), width, height, 3, png.data(), width * 3);
  printf("wrote %s\n", outPath.c_str());

  gpsDestroy(ctx);
  return 0;
}
