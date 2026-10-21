// Stub replacement for gaussian-point-splatting's src/core/input_manager.h.
//
// The upstream header pulls in GLFW. The only GPS translation unit that includes it
// and that we compile is world/camera.cpp (Camera::update reads mouse/keyboard input),
// which is never called by the Vulkan integration: the camera is driven by the host
// application. This header shadows the upstream one because gps_cuda puts the
// "stubs" directory first on the include path.
#pragma once
#include <glm/glm.hpp>

class InputManager
{
public:
  static void      update() {}
  static void      release() {}
  static glm::vec3 getMovementVector() { return glm::vec3(0.0f); }
  static glm::vec2 getMouseDelta() { return glm::vec2(0.0f); }
  static bool      isBackSpacePressed() { return false; }
  static bool      isScreenShotButtonPressed() { return false; }
  static int       getMouseScrollDelta() { return 0; }
};
