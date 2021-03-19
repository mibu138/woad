#include "lights.glsl"
#include "material.glsl"

layout(set = 0, binding = 0) uniform Camera {
    mat4 view;
    mat4 proj;
    mat4 xform;
} camera;

layout(push_constant) uniform PushConstant {
    layout(offset = 8) uint     lightCount;
} push;

