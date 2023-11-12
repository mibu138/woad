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

#define MODEL_PATH_1 "pome/assets/Box/glTF-Binary/Box.glb"
#define MODEL_PATH_2 "pome/assets/BoomBox/glTF-Binary/BoomBox.glb"

static void init_scene_prims(OnyxScene *scene)
{
    int err = 0;
    err = hell_gltf_init(MODEL_PATH_2, &gltf_data);

    assert(!err);

    err = hell_gltf_read_mesh(&gltf_data, orb.memory, &geo_template, &geo);

    assert(!err);

    hell_gltf_term(&gltf_data);

    onyx_scene_add_prim(scene, &geo, COAL_MAT4_IDENT, (OnyxMaterialHandle){0});
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
