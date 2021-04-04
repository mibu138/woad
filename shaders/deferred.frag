#version 460
#extension GL_GOOGLE_include_directive : enable

#include "common.glsl"
#include "frag-common.glsl"

layout(location = 0) in  vec2 uv;

layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 0, rgba32f) uniform image2D imageWorldP;
layout(set = 1, binding = 1, rgba32f) uniform image2D imageNormal;
layout(set = 1, binding = 2, rgba8)   uniform image2D imageAlbedo;
layout(set = 1, binding = 3, r16ui)   uniform uimage2D imageShadow;
layout(set = 1, binding = 4, r16)     uniform image2D imageRoughness;

void main()
{
    const ivec2 pixel = ivec2(gl_FragCoord.x, gl_FragCoord.y);
    const uint shadowMask = imageLoad(imageShadow, pixel).r;
    const vec3 P  = imageLoad(imageWorldP, pixel).xyz;
    const vec3 N  = imageLoad(imageNormal, pixel).xyz;
    const float roughness = imageLoad(imageRoughness, pixel).r;
    const vec3 Albedo = imageLoad(imageAlbedo, pixel).rgb;

    const vec3 campos   = vec3(camera.xform[3][0], camera.xform[3][1], camera.xform[3][2]);
    const vec3 ambient  = vec3(0.01);
    vec3 diffuse  = vec3(0);
    vec3 specular = vec3(0);

    for (int i = 0; i < push.lightCount; i++)
    {
        if ((shadowMask & (0x01 << i)) > 0)
        {
            vec3 eyeDir = normalize(campos - P);
            if (lights.light[i].type == DIR_LIGHT)
            {
                diffuse += lights.light[i].color * calcDiffuse(N, lights.light[i].vector) * lights.light[i].intensity;
                specular += lights.light[i].color * calcSpecular(N, lights.light[i].vector, eyeDir, SPEC_EXP) * lights.light[i].intensity;
            }
            else
            {
                vec3 dir      = normalize(P - lights.light[i].vector);
                float falloff =  1.0f / max(length(P - lights.light[i].vector), 0.001); // to prevent div by 0
                diffuse += lights.light[i].color * calcDiffuse(N, dir) * lights.light[i].intensity * falloff;
                specular += lights.light[i].color * calcSpecular(N, dir, eyeDir, SPEC_EXP) * lights.light[i].intensity * falloff;
            }
        }
    }

    specular = specular * (1 - roughness);
    vec3 illume = (diffuse + specular * 4);
    outColor = vec4(Albedo, 1) * vec4(illume + ambient, 1);
}

