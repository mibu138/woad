#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "material.glsl"

layout(location = 0) in       vec3 worldPos;
layout(location = 1) flat in  uint matId;

layout(location = 0) out vec4  outWorld;
layout(location = 1) out vec4  outNormal;
layout(location = 2) out vec4  outAlbedo;
layout(location = 3) out float outRoughness;

layout(set = 0, binding = 3) uniform sampler2D textures[];

layout(set = 0, binding = 4) uniform Materials {
    Material mat[14];
} materials;

void main()
{
    const Material mat = materials.mat[matId];
    outAlbedo = vec4(mat.r, mat.g, mat.b, 1);
    outRoughness = mat.roughness;
    outWorld  = vec4(worldPos, 1);
    vec3 tanget = dFdx(worldPos);
    vec3 bitang = dFdy(worldPos);
    vec3 N = normalize(cross(bitang, tanget));
    outNormal = vec4(N, 1); //wrong but for now...
}

