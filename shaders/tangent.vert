#version 460
#extension GL_GOOGLE_include_directive : enable

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 norm;
layout(location = 2) in vec3 tangent;
layout(location = 3) in float sign;
layout(location = 4) in vec2 uvw;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec2 outUv;
layout(location = 2) out uint outMatId;
layout(location = 3) out mat3 outTBN;

#include "vert-common.glsl"

void main()
{
    const mat4 xform = push.xform;
    const vec4 worldPos = xform * vec4(pos, 1.0);
    gl_Position = camera.proj * camera.view * worldPos;
    vec3 bitangent = sign * normalize(cross(norm, tangent));
    vec3 T = normalize(vec3(xform * vec4(tangent, 0)));
    vec3 B = normalize(vec3(xform * vec4(bitangent, 0)));
    vec3 N = normalize(vec3(xform * vec4(norm, 0)));

    outWorldPos = worldPos.xyz; 
    outUv = uvw.st;
    outMatId = push.matId;
    outTBN = mat3(T, B, N);
}
