#define MAX_LIGHTS 8
#define SPEC_EXP 128

#define POINT_LIGHT 0
#define DIR_LIGHT   1

struct Light {
    vec3  vector; // position or direction based on type
    float intensity;
    vec3  color;
    int   type;
};

layout(set = 0, binding = 2) uniform Lights {
    Light light[MAX_LIGHTS]; 
} lights;
