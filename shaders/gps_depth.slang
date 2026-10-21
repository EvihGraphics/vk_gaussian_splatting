/*
 * Depth pass of the Gaussian Point Splatting (CUDA) pipeline.
 *
 * Writes the per-pixel depth of the front-most splatted point, produced by gps_cuda.dll in
 * the depth plane of the shared output buffer (Vulkan [0,1] convention, row 0 = top), into
 * the hardware depth buffer of the G-buffer. This is what the raster pipelines provide for
 * the visual helpers (grid, gizmo), which test their fragments against the scene depth.
 */

struct GpsDepthPushConstants
{
  uint2 size;
  uint  depthOffset;  // offset of the depth plane, in floats
  uint  pad;
};

// clang-format off
[[vk::push_constant]] ConstantBuffer<GpsDepthPushConstants> pc;
[[vk::binding(0, 0)]] StructuredBuffer<float>               gpsOutput;
// clang-format on

struct VertexOutput
{
  float4 position : SV_Position;
};

// Fullscreen triangle
[shader("vertex")]
VertexOutput vertmain(uint vertexId : SV_VertexID)
{
  VertexOutput output;
  const float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
  output.position = float4(uv * 2.0f - 1.0f, 0.0f, 1.0f);
  return output;
}

struct FragmentOutput
{
  float depth : SV_Depth;
};

[shader("fragment")]
FragmentOutput fragmain(VertexOutput input)
{
  FragmentOutput output;
  const uint2    pixel = uint2(input.position.xy);
  if(pixel.x >= pc.size.x || pixel.y >= pc.size.y)
    discard;
  output.depth = gpsOutput[pc.depthOffset + pixel.y * pc.size.x + pixel.x];
  return output;
}
