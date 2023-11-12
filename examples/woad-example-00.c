#include "examplebase.h"

HellGltfData gltf_data;
OnyxGeometry geo;
OnyxGeometryTemplate geo_template = {
    .attribute_count = 3,
    .attribute_sizes = {12, 12, 8},
    .attribute_types = {ONYX_ATTRIBUTE_TYPE_POS, ONYX_ATTRIBUTE_TYPE_NORMAL, ONYX_ATTRIBUTE_TYPE_UV},
    .type = ONYX_GEOMETRY_TYPE_TRIANGLES,
    .flags = 0
};

typedef struct Model {
    const char *path;
    float      scale;
    float      y_rot;
} Model;

static const Model models[] = {
    { "pome/assets/Box/glTF-Binary/Box.glb", 1.0, 0.0 },
    { "pome/assets/BoomBox/glTF-Binary/BoomBox.glb", 50.0, COAL_PI },
};

#define MODEL_NUM 1

static void init_scene_prims(OnyxScene *scene)
{
    int err = 0;
    const Model model = models[MODEL_NUM];

    err = hell_gltf_init(model.path, &gltf_data);

    assert(!err);

    err = hell_gltf_read_mesh(&gltf_data, orb.memory, &geo_template, &geo);

    assert(!err);

    hell_gltf_term(&gltf_data);

    CoalMat4 xform = COAL_MAT4_IDENT;
    xform = coal_scale_uniform_mat4(model.scale, xform);
    xform = coal_rotate_y_mat4(model.y_rot, xform);

    onyx_scene_add_prim(scene, &geo, xform, (OnyxMaterialHandle){0});
}

int
main(int argc, char* argv[])
{
    ExampleInterface ex = {
        .init_scene_prims = init_scene_prims,
        .init_scene_lights = NULL,
    };
    example_main(&ex);
    return 0;
}
