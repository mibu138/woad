#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "material.glsl"

layout(location = 0) in       vec3 worldPos;
layout(location = 1) in       vec2 uv;
layout(location = 2) flat in  uint matId;
layout(location = 3) in       mat3 TBN;

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
    const vec2 st = vec2(uv.x, -uv.y + 1);
    const Material mat = materials.mat[matId];

    if (mat.textureAlbedo > 0)
        outAlbedo    = texture(textures[mat.textureAlbedo], st);
    else
        outAlbedo = vec4(mat.r, mat.g, mat.b, 1);

    if (mat.textureRoughness > 0)
        outRoughness = texture(textures[mat.textureRoughness], st).r;
    else
        outRoughness = mat.roughness;

    vec3 normal = vec3(0, 0, 1);
    if (mat.textureNormal > 0)
    {
        normal = texture(textures[mat.textureNormal], st).xyz;
        normal = normal * 2.0 - 1.0;
    }
    outNormal = vec4(normalize(TBN * normal), 1);

    outWorld  = vec4(worldPos, 1);
}

