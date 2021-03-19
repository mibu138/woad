#version 460
#extension GL_GOOGLE_include_directive : enable

layout(location = 0) in  vec2 uv;

layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 0, rgba32f) uniform image2D imageWorldP;
layout(set = 1, binding = 1, rgba32f) uniform image2D imageNormal;
layout(set = 1, binding = 2, r8)    uniform image2D imageShadow;

void main()
{
    const ivec2 pixel = ivec2(gl_FragCoord.x, gl_FragCoord.y);
    float shadow = imageLoad(imageShadow, pixel).r;
    vec3 P = imageLoad(imageWorldP, pixel).xyz;
    vec3 N = imageLoad(imageNormal, pixel).xyz;
    vec3 C = N * shadow;
    outColor = vec4(C, 1.0);
}

