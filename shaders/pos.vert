#version 460
#extension GL_GOOGLE_include_directive : enable

#include "vert-common.glsl"

layout(location = 0) in vec3 pos;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out uint outMatId;

void main()
{
    vec4 worldPos = model.xform[push.primId] * vec4(pos, 1.0);
    gl_Position = camera.proj * camera.view * worldPos;
    outWorldPos = worldPos.xyz; 
    outMatId = push.matId;
}
