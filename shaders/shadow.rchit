#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "shadow-common.glsl"

layout(location = 0) rayPayloadInEXT hitPayload payload;

void main()
{
    payload.illume = 0x0;
}
