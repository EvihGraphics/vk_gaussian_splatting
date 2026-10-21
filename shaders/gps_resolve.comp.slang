/*
 * Resolve pass of the Gaussian Point Splatting (CUDA) pipeline.
 *
 * Copies the RGBA float image written by gps_cuda.dll into an external-memory buffer
 * (row 0 = top) into the G-buffer color image the rest of the sample consumes
 * (temporal accumulation, tone mapping, image comparison, presentation).
 */

struct GpsResolvePushConstants
{
  uint2 size;
};

// clang-format off
[[vk::push_constant]]       ConstantBuffer<GpsResolvePushConstants> pc;
[[vk::binding(0, 0)]]       StructuredBuffer<float4>                srcColor;
[[vk::binding(1, 0)]]       RWTexture2D<float4>                     dstImage;
// clang-format on

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
  const uint2 pixel = dispatchThreadID.xy;
  if(pixel.x >= pc.size.x || pixel.y >= pc.size.y)
    return;

  dstImage[pixel] = srcColor[pixel.y * pc.size.x + pixel.x];
}
