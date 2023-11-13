#include "examplebase.h"

HellGltfData gltf_data;
OnyxGeometry geo;
OnyxGeometry ground;
OnyxPrimitiveHandle geo_prim;
OnyxPrimitiveHandle ground_prim;
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

    ground = onyx_create_cube(orb.memory, false);
    CoalMat4 ground_xform = COAL_MAT4_IDENT;
    ground_xform = coal_translate_mat4((CoalVec3){0, -1, 0}, ground_xform);
    ground_xform = coal_scale_non_uniform_mat4((CoalVec3){1.5, 1.0, 1.5}, ground_xform);

    OnyxMaterialHandle material = onyx_scene_create_material(
            scene, (CoalVec3){0.9, 0.7, 0.1}, 0.8, 
            (OnyxTextureHandle){0},
            (OnyxTextureHandle){0},
            (OnyxTextureHandle){0});

    OnyxMaterialHandle ground_material = onyx_scene_create_material(
            scene, (CoalVec3){0.6, 0.6, 0.62}, 0.9, 
            (OnyxTextureHandle){0},
            (OnyxTextureHandle){0},
            (OnyxTextureHandle){0});

    geo_prim = onyx_scene_add_prim(scene, &geo, xform, material);
    ground_prim = onyx_scene_add_prim(scene, &ground, ground_xform, ground_material);
}

static void update_scene(OnyxScene *s, i64 fi, i64 dt)
{
    CoalMat4 xform = coal_rotate_y_mat4(0.01, COAL_MAT4_IDENT);
    onyx_update_prim_xform(s, geo_prim, xform);
}

int
main(int argc, char* argv[])
{
    ExampleInterface ex = {
        .init_scene_prims = init_scene_prims,
        .init_scene_lights = NULL,
        .enable_ray_tracing = true,
        .update_scene = update_scene,
    };
    woad_example_main(&ex);
    return 0;
}
