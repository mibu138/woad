#version 460
#extension GL_GOOGLE_include_directive : enable

#include "vert-common.glsl"

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 norm;
layout(location = 2) in vec2 uvw;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outUv;
layout(location = 3) out uint outMatId;

void main()
{
    vec4 worldPos = push.xform * vec4(pos, 1.0);
    gl_Position = camera.proj * camera.view * worldPos;
    outWorldPos = worldPos.xyz; 
    outNormal = normalize((push.xform * vec4(norm, 0.0)).xyz); // this is fine as long as we only allow uniform scales
    outUv = uvw.st;
    outMatId = push.matId;
}
