layout(set = 0, binding = 0) uniform Camera {
    mat4 view;
    mat4 proj;
    mat4 xform;
} camera;

layout (constant_id = 1) const int MAX_PRIMS = 2000;

layout(set = 0, binding = 1) uniform Model {
    mat4 xform[MAX_PRIMS];
} model;

layout(push_constant) uniform PushConstant {
    uint primId;
    uint matId;
} push;
