#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "common.glsl"
#include "frag-common.glsl"

layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec2 uv;
layout(location = 2) in mat3 TBN;

layout(location = 0) out vec4 outColor;

void main()
{
    const vec2 st = vec2(uv.x, uv.y * -1 + 1);
    const vec4 albedo     = texture(textures[push.material.textureAlbedo], st);
    const float roughness = texture(textures[push.material.textureRoughness], st).r * push.material.roughness;
    vec3 normal           = texture(textures[push.material.textureNormal], st).rgb;
    normal = normal * 2.0 - 1.0;
    normal = normalize(TBN * normal);
    const vec3 campos   = vec3(camera.xform[3][0], camera.xform[3][1], camera.xform[3][2]);
    const vec3 ambient  = vec3(0.01);
    vec3 diffuse  = vec3(0);
    vec3 specular = vec3(0);
    for (int i = 0; i < push.lightCount; i++)
    {
        vec3 eyeDir = normalize(campos - worldPos);
        if (lights.light[i].type == 1)
        {
            diffuse += lights.light[i].color * calcDiffuse(normal, lights.light[i].vector) * lights.light[i].intensity;
            specular += lights.light[i].color * calcSpecular(normal, lights.light[i].vector, eyeDir, SPEC_EXP) * lights.light[i].intensity;
        }
        else
        {
            vec3 dir      = normalize(worldPos - lights.light[i].vector);
            float falloff =  1.0f / max(length(worldPos - lights.light[i].vector), 0.001); // to prevent div by 0
            diffuse += lights.light[i].color * calcDiffuse(normal, dir) * lights.light[i].intensity * falloff;
            specular += lights.light[i].color * calcSpecular(normal, dir, eyeDir, SPEC_EXP) * lights.light[i].intensity * falloff;
        }
    }
    specular = specular * (1 - roughness);
    vec3 illume = diffuse + specular * 4;
    outColor = vec4(albedo.rgb * push.material.color, 1) * vec4(illume + ambient, 1);
}
