#define COAL_SIMPLE_TYPE_NAMES
#define ONYX_SIMPLE_TYPE_NAMES
#include "woad.h"

#include <assert.h>
#include <coal/coal.h>
#include <hell/hell.h>
#include <hell/len.h>
#include <hell/io.h>
#include <memory.h>
#include <onyx/attribute.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef OnyxCommand               Command;
typedef OnyxImage                 Image;
typedef OnyxLight                 Light;
typedef OnyxMaterial              Material;
typedef OnyxBuffer                BufferRegion;
typedef OnyxAccelerationStructure AccelerationStructure;
typedef OnyxPrimitive             Prim;

#define WOAD_SPV_PREFIX "build/shaders/woad_shaders"
#define ONYX_SPV_PREFIX "build/pome/src/onyx/shaders/onyx_shaders"

// quick hack
#define ONYX_S_MAX_MATERIALS 10

typedef OnyxMask AttrMask;

enum {
    POS_BIT    = 1 << 0,
    NORMAL_BIT = 1 << 1,
    UV_BIT     = 1 << 2,
    TAN_BIT    = 1 << 3,
    SIGN_BIT   = 1 << 4,
};

#define POS_NOR_UV_TAN_MASK (POS_BIT | NORMAL_BIT | UV_BIT | TAN_BIT | SIGN_BIT)
#define POS_NOR_UV_MASK (POS_BIT | NORMAL_BIT | UV_BIT)
#define POS_MASK (POS_BIT)

enum {
    PIPE_LAYOUT_MAIN,
};

enum { DESC_SET_MAIN, DESC_SET_DEFERRED, DESC_SET_COUNT };

enum {
    PIPELINE_GBUFFER_POS_NOR_UV,
    PIPELINE_GBUFFER_POS_NOR_UV_TAN,
    PIPELINE_GBUFFER_POS,
    GBUFFER_PIPELINE_COUNT
};

_Static_assert(GBUFFER_PIPELINE_COUNT < ONYX_MAX_PIPELINES,
               "GRAPHICS_PIPELINE_COUNT must be less than ONYX_MAX_PIPELINES");

#define MAX_PRIM_COUNT ONYX_S_MAX_PRIMS
#define MAX_LIGHT_COUNT 16

// TODO: This is what we need to initialize....
typedef struct {
    Light elems[MAX_LIGHT_COUNT];
} Lights;

// we may be able to use this space to update certain prim transforms
// faster than running through the whole list of them
// especially when dealing with many hundreds of prims.
// currently though this is not used.
typedef struct {
    Mat4 xform[16];
} Xforms;

typedef struct {
    Mat4 view;
    Mat4 proj;
    Mat4 camera;
} Camera;

#define MAX_FRAMES_IN_FLIGHT 2

define_array_type(AccelerationStructure, accel_struct);

static VkRenderPass gbufferRenderPass;
static VkRenderPass deferredRenderPass;
static uint32_t     graphic_queue_family_index;

static VkFramebuffer gframebuffer;
static VkFramebuffer swapImageBuffer[MAX_FRAMES_IN_FLIGHT];

static VkPipeline gbufferPipelines[GBUFFER_PIPELINE_COUNT];
static VkPipeline defferedPipeline;

static VkPipeline              raytracePipeline;
static OnyxShaderBindingTable shaderBindingTable;

static BufferRegion cameraBuffers[MAX_FRAMES_IN_FLIGHT];
static BufferRegion xformsBuffers[MAX_FRAMES_IN_FLIGHT];
static BufferRegion lightsBuffers[MAX_FRAMES_IN_FLIGHT];
static BufferRegion materialsBuffers[MAX_FRAMES_IN_FLIGHT];

static const OnyxInstance* instance;
static OnyxMemory*         memory;
static VkDevice             device;

static OnyxPrimitiveList pipelinePrimLists[GBUFFER_PIPELINE_COUNT];

// raytrace stuff

static AccelerationStructureArray blas_array;
static AccelerationStructure tlas[MAX_FRAMES_IN_FLIGHT];

// raytrace stuff

static VkDescriptorSetLayout descriptorSetLayouts[DESC_SET_COUNT];
static VkDescriptorPool      descriptorPool;
static VkDescriptorSet       descriptorSets[MAX_FRAMES_IN_FLIGHT][2];

static VkPipelineLayout pipelineLayout;

static Image renderTargetDepth;
static Image imageWorldP;
static Image imageNormal;
static Image imageShadow;
static Image imageAlbedo;
static Image imageRoughness;

static const VkFormat depthFormat  = VK_FORMAT_D32_SFLOAT;
static const VkFormat formatImageP = VK_FORMAT_R32G32B32A32_SFLOAT;
static const VkFormat formatImageN = VK_FORMAT_R32G32B32A32_SFLOAT;
static const VkFormat formatImageShadow =
    VK_FORMAT_R16_UINT; // maximum of 16 lights.
static const VkFormat formatImageAlbedo    = VK_FORMAT_R8G8B8A8_UNORM;
static const VkFormat formatImageRoughness = VK_FORMAT_R32_SFLOAT;

// declarations for overview and navigation
static void initDescriptorSetsAndPipelineLayouts(void);
static void updateDescriptors(void);
static void syncScene(const uint32_t frameIndex);

static bool raytracing_disabled = false;

void r_InitRenderer(const OnyxScene* scene_, VkImageLayout finalImageLayout,
                    bool openglStyle);
void r_CleanUp(void);
uint8_t
r_GetMaxFramesInFlight(void)
{
    return MAX_FRAMES_IN_FLIGHT;
}

static void
initAttachments(uint32_t windowWidth, uint32_t windowHeight)
{
    renderTargetDepth = onyx_create_image(
        memory, windowWidth, windowHeight, VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT, VK_SAMPLE_COUNT_1_BIT, 1,
        ONYX_MEMORY_DEVICE_TYPE);

    imageWorldP = onyx_create_image(
        memory, windowWidth, windowHeight, formatImageP,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, VK_SAMPLE_COUNT_1_BIT, 1,
        ONYX_MEMORY_DEVICE_TYPE);

    imageNormal = onyx_create_image(
        memory, windowWidth, windowHeight, formatImageN,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, VK_SAMPLE_COUNT_1_BIT, 1,
        ONYX_MEMORY_DEVICE_TYPE);

    imageShadow = onyx_create_image(
        memory, windowWidth, windowHeight, formatImageShadow,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, VK_SAMPLE_COUNT_1_BIT, 1,
        ONYX_MEMORY_DEVICE_TYPE);

    imageAlbedo = onyx_create_image(
        memory, windowWidth, windowHeight, formatImageAlbedo,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, VK_SAMPLE_COUNT_1_BIT, 1,
        ONYX_MEMORY_DEVICE_TYPE);

    imageRoughness = onyx_create_image(
        memory, windowWidth, windowHeight, formatImageRoughness,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, VK_SAMPLE_COUNT_1_BIT, 1,
        ONYX_MEMORY_DEVICE_TYPE);

    OnyxCommandPool pool =
        onyx_create_command_pool_(device, graphic_queue_family_index,
                               VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, 1);
    VkCommandBuffer cmdbuf = pool.cmdbufs[0];
    onyx_begin_command_buffer(cmdbuf);

    OnyxBarrierScopes b  = {};
    b.src.access_mask = 0;
    b.dst.access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.src.stage_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    b.dst.stage_mask = VK_PIPELINE_STAGE_TRANSFER_BIT;

    onyx_cmd_transition_image_layout(cmdbuf, b, VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_GENERAL, 1,
                                  imageShadow.handle);

    onyx_cmd_clear_color_image(cmdbuf, imageShadow.handle, VK_IMAGE_LAYOUT_GENERAL,
                            0, 1, 1.0, 0, 0, 0);

    onyx_end_command_buffer(cmdbuf);

    VkSubmitInfo si = onyx_submit_info(0, NULL, NULL, 1, &cmdbuf, 0, NULL);

    VkQueue queue = onyx_get_graphics_queue(instance, 0);
    VkFence fence;
    onyx_create_fence(device, false, &fence);
    vkQueueSubmit(queue, 1, &si, fence);
    onyx_wait_for_fence(device, &fence);

    onyx_destroy_fence(device, fence);
    onyx_destroy_command_pool(device, &pool);
}

static void
initGbufRenderPass(void)
{
    VkAttachmentDescription attachmentWorldP = {
        .flags          = 0,
        .format         = formatImageP,
        .samples        = VK_SAMPLE_COUNT_1_BIT,
        .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout    = VK_IMAGE_LAYOUT_GENERAL};

    VkAttachmentDescription attachmentNormal = {
        .flags          = 0,
        .format         = formatImageN,
        .samples        = VK_SAMPLE_COUNT_1_BIT,
        .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout    = VK_IMAGE_LAYOUT_GENERAL};

    VkAttachmentDescription attachmentAlbedo = {
        .flags          = 0,
        .format         = formatImageAlbedo,
        .samples        = VK_SAMPLE_COUNT_1_BIT,
        .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout    = VK_IMAGE_LAYOUT_GENERAL};

    VkAttachmentDescription attachmentRoughness = {
        .flags          = 0,
        .format         = formatImageRoughness,
        .samples        = VK_SAMPLE_COUNT_1_BIT,
        .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout    = VK_IMAGE_LAYOUT_GENERAL};

    VkAttachmentDescription attachmentDepth = {
        .flags          = 0,
        .format         = depthFormat,
        .samples        = VK_SAMPLE_COUNT_1_BIT,
        .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkAttachmentReference refWorldP = {
        .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkAttachmentReference refNormal = {
        .attachment = 1, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkAttachmentReference refAlbedo = {
        .attachment = 2, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkAttachmentReference refRough = {
        .attachment = 3, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkAttachmentReference refDepth = {
        .attachment = 4,
        .layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkAttachmentReference colorRefs[] = {refWorldP, refNormal, refAlbedo,
                                         refRough};

    VkSubpassDescription subpass = {.flags = 0,
                                    .pipelineBindPoint =
                                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    .inputAttachmentCount    = 0,
                                    .pInputAttachments       = NULL,
                                    .colorAttachmentCount    = LEN(colorRefs),
                                    .pColorAttachments       = colorRefs,
                                    .pResolveAttachments     = NULL,
                                    .pDepthStencilAttachment = &refDepth,
                                    .preserveAttachmentCount = 0,
                                    .pPreserveAttachments    = NULL};

    VkSubpassDependency dep1 = {
        .srcSubpass   = VK_SUBPASS_EXTERNAL,
        .dstSubpass   = 0,
        .srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        .dstStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, // may not be
                                                        // necesary
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT};

    VkSubpassDependency dep2 = {
        .srcSubpass = 0,
        .dstSubpass = VK_SUBPASS_EXTERNAL,
        .srcStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, // may not be
                                                        // necesary
        .dstStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        .srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dstAccessMask   = VK_ACCESS_SHADER_READ_BIT,
        .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT};

    VkSubpassDependency deps[] = {dep1, dep2};

    VkAttachmentDescription attachments[] = {
        attachmentWorldP, attachmentNormal, attachmentAlbedo,
        attachmentRoughness, attachmentDepth};

    VkRenderPassCreateInfo rpiInfo = {
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext           = NULL,
        .flags           = 0,
        .attachmentCount = LEN(attachments),
        .pAttachments    = attachments,
        .subpassCount    = 1,
        .pSubpasses      = &subpass,
        .dependencyCount = LEN(deps),
        .pDependencies   = deps};

    V_ASSERT(vkCreateRenderPass(device, &rpiInfo, NULL, &gbufferRenderPass));
}

static void
initGbufferFramebuffer(u32 w, u32 h)
{
    const VkImageView attachments[] = {imageWorldP.view, imageNormal.view,
                                       imageAlbedo.view, imageRoughness.view,
                                       renderTargetDepth.view};

    const VkFramebufferCreateInfo fbi = {
        .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext           = NULL,
        .flags           = 0,
        .renderPass      = gbufferRenderPass,
        .attachmentCount = 5,
        .pAttachments    = attachments,
        .width           = w,
        .height          = h,
        .layers          = 1};

    V_ASSERT(vkCreateFramebuffer(device, &fbi, NULL, &gframebuffer));
}

static void
initSwapFramebuffer(const WoadFrame* frame)
{
    uint32_t                      windowWidth  = frame->width;
    uint32_t                      windowHeight = frame->height;
    const VkFramebufferCreateInfo fbi          = {
                 .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                 .pNext           = NULL,
                 .flags           = 0,
                 .renderPass      = deferredRenderPass,
                 .attachmentCount = 1,
                 .pAttachments    = &frame->view,
                 .width           = windowWidth,
                 .height          = windowHeight,
                 .layers          = 1,
    };

    V_ASSERT(vkCreateFramebuffer(device, &fbi, NULL,
                                 &swapImageBuffer[frame->index]));
}

static void
initDescriptorSetsAndPipelineLayouts(void)
{
    OnyxDescriptor bindings0[] = {
        {// camera
         .count = 1,
         .type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .stages =
             VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
        {// xforms
         .count = 1,
         .type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .stages      = VK_SHADER_STAGE_VERTEX_BIT},
        {// lights
         .count = 1,
         .type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .stages =
             VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        {                       // textures
         .count = 10, // because this is an array of samplers. others
                                // are structs of arrays.
         .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .stages      = VK_SHADER_STAGE_FRAGMENT_BIT,
         .binding_flags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},
        {
            // materials
            .count = 1, // because this is an array of samplers.
                                  // others are structs of arrays.
            .type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .stages      = VK_SHADER_STAGE_FRAGMENT_BIT,
        }};

    OnyxDescriptor bindings1[] = {
        {
            // worldp storage image
            .count = 1,
            .type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .stages =
                VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        },
        {
            // normal storage image
            .count = 1,
            .type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .stages =
                VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        },
        {// albedo storage image
         .count = 1,
         .type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .stages      = VK_SHADER_STAGE_FRAGMENT_BIT},
        {
            // shadow storage image
            .count = 1,
            .type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .stages =
                VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        },
        {// roughness storage image
         .count = 1,
         .type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .stages      = VK_SHADER_STAGE_FRAGMENT_BIT},
        {
            // top level AS
            .count = 1,
            .type            = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            .stages      = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        }};

    onyx_create_descriptor_set_layout(device, LEN(bindings0),
                                      bindings0, &descriptorSetLayouts[0]);

    onyx_create_descriptor_set_layout(device, LEN(bindings1),
                                      bindings1, &descriptorSetLayouts[1]);

    OnyxDescriptorPoolParms pool_parms = {
        .accelerationStructureCount = 20,
        .combinedImageSamplerCount = 24,
        .storageImageCount = 20,
        .dynamicUniformBufferCount = 10,
        .storageImageCount = 20,
        .uniformBufferCount = 20,
        .inputAttachmentCount = 0,
    };

    onyx_create_descriptor_pool(device, pool_parms, &descriptorPool);

    onyx_allocate_descriptor_sets(device, descriptorPool, DESC_SET_COUNT, descriptorSetLayouts, descriptorSets[0]);
    onyx_allocate_descriptor_sets(device, descriptorPool, DESC_SET_COUNT, descriptorSetLayouts, descriptorSets[1]);

    const VkPushConstantRange pcPrimId = {
        .offset = 0,
        .size   = sizeof(Mat4) + sizeof(uint32_t) * 2, // prim id, material id
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT};

    // light count
    const VkPushConstantRange pcFrag = {
        .offset = sizeof(Mat4) + sizeof(uint32_t) * 2, // prim id, material id
        .size   = sizeof(uint32_t),
        .stageFlags =
            VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR};

    const VkPushConstantRange ranges[] = {pcPrimId, pcFrag};

    const OnyxPipelineLayoutInfo pipeLayoutInfos[] = {
        {.descriptor_set_count = DESC_SET_COUNT,
         .descriptor_set_layouts = descriptorSetLayouts,
         .push_constant_count = LEN(ranges),
         .push_constant_ranges = ranges}};

    onyx_create_pipeline_layouts(device, 1, pipeLayoutInfos, &pipelineLayout);
}

static void
initPipelines(bool openglStyle)
{
    const OnyxGeoAttributeSize posAttrSizes[]          = {12};
    const OnyxGeoAttributeSize posNormalUvAttrSizes[3] = {12, 12, 8};
    const OnyxGeoAttributeSize tanAttrSizes[5] = {12, 12, 12, 4, 8};

    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                      VK_DYNAMIC_STATE_SCISSOR};

    VkSpecializationMapEntry mapEntry = {
        .constantID = 0, .offset = 0, .size = 4};

    int sign = openglStyle ? -1 : 1;

    VkSpecializationInfo fragSpecInfo = {.dataSize      = 4,
                                         .mapEntryCount = 1,
                                         .pData         = &sign,
                                         .pMapEntries   = &mapEntry};

    VkFrontFace frontface = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    OnyxVertexDescription tangent_vert_description = {
        .attribute_count = 5,
        .attribute_descriptions = {
            // pos
            { .location = 0, .binding = 0, .format = ONYX_VERT_POS_FORMAT, .offset = 0 },
            // normal
            { .location = 1, .binding = 1, .format = ONYX_VERT_POS_FORMAT, .offset = 0 },
            // tangent
            { .location = 2, .binding = 2, .format = ONYX_VERT_POS_FORMAT, .offset = 0 },
            // sign
            { .location = 3, .binding = 3, .format = VK_FORMAT_R32_SFLOAT, .offset = 0 },
            // uvs
            { .location = 4, .binding = 4, .format = VK_FORMAT_R32G32_SFLOAT, .offset = 0 },
        },
        .binding_count = 5,
        .binding_descriptions = {
            { .binding = 0, .stride = 12, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX },
            { .binding = 1, .stride = 12, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX },
            { .binding = 2, .stride = 12, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX },
            { .binding = 3, .stride = 4, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX },
            { .binding = 4, .stride = 8, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX },
        }
    };

    VkVertexInputBindingDescription b[] = {
        { .binding = 0, .stride = posNormalUvAttrSizes[0], .inputRate = VK_VERTEX_INPUT_RATE_VERTEX },
        { .binding = 1, .stride = posNormalUvAttrSizes[1], .inputRate = VK_VERTEX_INPUT_RATE_VERTEX },
        { .binding = 2, .stride = posNormalUvAttrSizes[2], .inputRate = VK_VERTEX_INPUT_RATE_VERTEX },
    };

    VkVertexInputAttributeDescription a[] = {
        { .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .location = 0, .offset = 0 },
        { .binding = 1, .format = VK_FORMAT_R32G32B32_SFLOAT, .location = 1, .offset = 0 },
        { .binding = 2, .format = VK_FORMAT_R32G32_SFLOAT, .location = 2, .offset = 0 },
    };

    VkVertexInputBindingDescription tan_b[] = {
        { .binding = 0, .stride = tanAttrSizes[0], .inputRate = VK_VERTEX_INPUT_RATE_VERTEX },
        { .binding = 1, .stride = tanAttrSizes[1], .inputRate = VK_VERTEX_INPUT_RATE_VERTEX },
        { .binding = 2, .stride = tanAttrSizes[2], .inputRate = VK_VERTEX_INPUT_RATE_VERTEX },
        { .binding = 3, .stride = tanAttrSizes[3], .inputRate = VK_VERTEX_INPUT_RATE_VERTEX },
        { .binding = 4, .stride = tanAttrSizes[4], .inputRate = VK_VERTEX_INPUT_RATE_VERTEX },
    };

    VkVertexInputAttributeDescription tan_a[] = {
        { .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .location = 0, .offset = 0 },
        { .binding = 1, .format = VK_FORMAT_R32G32B32_SFLOAT, .location = 1, .offset = 0 },
        { .binding = 2, .format = VK_FORMAT_R32G32B32_SFLOAT, .location = 2, .offset = 0 },
        { .binding = 3, .format = VK_FORMAT_R32_SFLOAT, .location = 3, .offset = 0 },
        { .binding = 4, .format = VK_FORMAT_R32G32_SFLOAT, .location = 4, .offset = 0 },
    };


    int       err = 0;
    ByteArray reg_vert_code, gbuffer_frag_code, tan_vert_code,
        tan_gbuffer_frag_code, pos_vert_code, gbuffer_pos_code, full_screen_vert_code, deferred_code;

    err |= hell_read_file(WOAD_SPV_PREFIX "/regular.vert.spv", &reg_vert_code);
    err |= hell_read_file(WOAD_SPV_PREFIX "/gbuffer.frag.spv", &gbuffer_frag_code);
    err |= hell_read_file(WOAD_SPV_PREFIX "/tangent.vert.spv", &tan_vert_code);
    err |= hell_read_file(WOAD_SPV_PREFIX "/gbuffertan.frag.spv", &tan_gbuffer_frag_code);
    err |= hell_read_file(WOAD_SPV_PREFIX "/pos.vert.spv", &pos_vert_code);
    err |= hell_read_file(WOAD_SPV_PREFIX "/gbufferpos.frag.spv", &gbuffer_pos_code);
    err |= hell_read_file(ONYX_SPV_PREFIX "/full-screen.vert.spv", &full_screen_vert_code);
    err |= hell_read_file(WOAD_SPV_PREFIX "/deferred.frag.spv", &deferred_code);

    if (err)
        fatal_error("Error reading spv files.");

    OnyxShaderInfo shader_stages_reg[] = {
         {
            .byte_count = reg_vert_code.count,
            .code = (void*)reg_vert_code.elems,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .entry_point = "main",
        },{
            .byte_count = gbuffer_frag_code.count,
            .code =(void*) gbuffer_frag_code.elems,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .entry_point = "main",
    }};

    OnyxShaderInfo shader_stages_tan[] = {
         {
            .byte_count = tan_vert_code.count,
            .code =(void*) tan_vert_code.elems,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .entry_point = "main",
        },{
            .byte_count = tan_gbuffer_frag_code.count,
            .code =(void*) tan_gbuffer_frag_code.elems,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .entry_point = "main",
    }};

    OnyxShaderInfo shader_stages_pos[] = {
         {
            .byte_count = pos_vert_code.count,
            .code =(void*) pos_vert_code.elems,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .entry_point = "main",
        },{
            .byte_count = gbuffer_pos_code.count,
            .code =(void*) gbuffer_pos_code.elems,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .entry_point = "main",
    }};

    OnyxShaderInfo shader_stages_deferred[] = {
         {
            .byte_count = full_screen_vert_code.count,
            .code =(void*) full_screen_vert_code.elems,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .entry_point = "main",
        },{
            .byte_count = deferred_code.count,
            .code =(void*) deferred_code.elems,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .entry_point = "main",
    }};

    OnyxPipelineColorBlendAttachment no_blend = {
        .blend_enable = false,
        .blend_mode = ONYX_BLEND_MODE_OVER,
    };

    OnyxPipelineColorBlendAttachment attachment_blends[4] = {
        no_blend,
        no_blend,
        no_blend,
        no_blend,
    };

    const OnyxGraphicsPipelineInfo gPipelineInfos[] = {
        (OnyxGraphicsPipelineInfo){
            .render_pass                      = gbufferRenderPass,
            .topology                         = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .layout                           = pipelineLayout,
            .rasterization_samples            = VK_SAMPLE_COUNT_1_BIT,
            .front_face                       = frontface,
            .attachment_count                 = 4,
            .attachment_blends                = attachment_blends,
            .line_width                       = 1.0,
            .depth_test_enable                = true,
            .depth_write_enable               = true,
            .vertex_binding_description_count = LEN(b),
            .vertex_binding_descriptions      = b,
            .vertex_attribute_description_count = LEN(a),
            .vertex_attribute_descriptions      = a,
            .dynamic_state_count                = LEN(dynamicStates),
            .dynamic_states                     = dynamicStates,
            .shader_stage_count                 = LEN(shader_stages_reg),
            .shader_stages                      = shader_stages_reg,
        },
        (OnyxGraphicsPipelineInfo){
            .render_pass           = gbufferRenderPass,
            .topology              = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .layout                = pipelineLayout,
            .rasterization_samples = VK_SAMPLE_COUNT_1_BIT,
            .front_face            = frontface,
            .line_width                       = 1.0,
            .depth_test_enable                = true,
            .depth_write_enable               = true,
            .attachment_count      = 4,
            .attachment_blends                = attachment_blends,
            .dynamic_state_count   = LEN(dynamicStates),
            .dynamic_states        = dynamicStates,
            .vertex_binding_description_count = LEN(tan_b),
            .vertex_binding_descriptions      = tan_b,
            .vertex_attribute_description_count = LEN(tan_a),
            .vertex_attribute_descriptions      = tan_a,
            .shader_stage_count    = LEN(shader_stages_tan),
            .shader_stages         = shader_stages_tan,
        },
        (OnyxGraphicsPipelineInfo){
            .render_pass                      = gbufferRenderPass,
            .topology                         = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .layout                           = pipelineLayout,
            .rasterization_samples            = VK_SAMPLE_COUNT_1_BIT,
            .line_width                       = 1.0,
            .depth_test_enable                = true,
            .depth_write_enable               = true,
            .front_face                       = frontface,
            .attachment_count                 = 4,
            .attachment_blends                = attachment_blends,
            .dynamic_state_count              = LEN(dynamicStates),
            .dynamic_states                   = dynamicStates,
            .vertex_binding_description_count = LEN(b),
            .vertex_binding_descriptions      = b,
            .vertex_attribute_description_count = LEN(a),
            .vertex_attribute_descriptions      = a,
            .shader_stage_count                 = LEN(shader_stages_pos),
            .shader_stages                      = shader_stages_pos,
        }};

    const OnyxGraphicsPipelineInfo defferedPipeInfo = {
        .render_pass = deferredRenderPass,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .layout            = pipelineLayout,
        .rasterization_samples = VK_SAMPLE_COUNT_1_BIT,
        .front_face = VK_FRONT_FACE_CLOCKWISE,
        .attachment_count = 1,
        .attachment_blends = attachment_blends,
        .dynamic_state_count = LEN(dynamicStates),
        .dynamic_states = dynamicStates,
        .shader_stage_count = LEN(shader_stages_deferred),
        .shader_stages = shader_stages_deferred,
            .line_width                       = 1.0,
    };

    const OnyxRayTracePipelineInfo rtPipelineInfo = {
        .layout        = pipelineLayout,
        .raygen_count    = 1,
        .raygen_shaders = (char*[]){WOAD_SPV_PREFIX "/shadow.rgen.spv"},
        .miss_count      = 1,
        .miss_shaders = (char*[]){WOAD_SPV_PREFIX "/shadow.rmiss.spv"},
        .chit_count = 1,
        .chit_shaders = (char*[]){WOAD_SPV_PREFIX "/shadow.rchit.spv"}};

    assert(LEN(gPipelineInfos) == GBUFFER_PIPELINE_COUNT);

    onyx_create_graphics_pipelines(device, LEN(gPipelineInfos), gPipelineInfos,
                                 gbufferPipelines);
    onyx_create_graphics_pipelines(device, 1, &defferedPipeInfo,
                                 &defferedPipeline);
    if (!raytracing_disabled)
        onyx_create_ray_trace_pipelines(device, memory, 1, &rtPipelineInfo,
                                     &raytracePipeline, &shaderBindingTable);
}

static void
updateGbufferDescriptors(void)
{
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        VkDescriptorImageInfo worldPInfo = {.sampler   = imageWorldP.sampler,
                                            .imageView = imageWorldP.view,
                                            .imageLayout =
                                                VK_IMAGE_LAYOUT_GENERAL};

        VkDescriptorImageInfo normalInfo = {.sampler   = imageNormal.sampler,
                                            .imageView = imageNormal.view,
                                            .imageLayout =
                                                VK_IMAGE_LAYOUT_GENERAL};

        VkDescriptorImageInfo albedoInfo = {.sampler   = imageAlbedo.sampler,
                                            .imageView = imageAlbedo.view,
                                            .imageLayout =
                                                VK_IMAGE_LAYOUT_GENERAL};

        VkDescriptorImageInfo shadowInfo = {.sampler   = imageShadow.sampler,
                                            .imageView = imageShadow.view,
                                            .imageLayout =
                                                VK_IMAGE_LAYOUT_GENERAL};

        VkDescriptorImageInfo roughnessInfo = {
            .sampler     = imageRoughness.sampler,
            .imageView   = imageRoughness.view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

        VkWriteDescriptorSet writes[] = {
            {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
             .dstArrayElement = 0,
             .dstSet     = descriptorSets[i][DESC_SET_DEFERRED],
             .dstBinding = 0,
             .descriptorCount = 1,
             .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
             .pImageInfo      = &worldPInfo},
            {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
             .dstArrayElement = 0,
             .dstSet     = descriptorSets[i][DESC_SET_DEFERRED],
             .dstBinding = 1,
             .descriptorCount = 1,
             .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
             .pImageInfo      = &normalInfo},
            {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
             .dstArrayElement = 0,
             .dstSet     = descriptorSets[i][DESC_SET_DEFERRED],
             .dstBinding = 2,
             .descriptorCount = 1,
             .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
             .pImageInfo      = &albedoInfo},
            {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
             .dstArrayElement = 0,
             .dstSet     = descriptorSets[i][DESC_SET_DEFERRED],
             .dstBinding = 3,
             .descriptorCount = 1,
             .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
             .pImageInfo      = &shadowInfo},
            {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
             .dstArrayElement = 0,
             .dstSet     = descriptorSets[i][DESC_SET_DEFERRED],
             .dstBinding = 4,
             .descriptorCount = 1,
             .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
             .pImageInfo      = &roughnessInfo}};

        vkUpdateDescriptorSets(device, LEN(writes), writes, 0, NULL);
    }
}

static void
updateASDescriptors(int frame_index)
{
    VkWriteDescriptorSetAccelerationStructureKHR asInfo = {
        .sType =
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .accelerationStructureCount = 1,
        .pAccelerationStructures    = &tlas[frame_index].handle};

    VkWriteDescriptorSet writeDS = {
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstArrayElement = 0,
        .dstSet     = descriptorSets[frame_index][DESC_SET_DEFERRED],
        .dstBinding = 5,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        .pNext           = &asInfo};

    vkUpdateDescriptorSets(device, 1, &writeDS, 0, NULL);
}

static void
updateDescriptors(void)
{
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        // camera creation
        cameraBuffers[i] = onyx_request_buffer_region(
            memory, sizeof(Camera), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            ONYX_MEMORY_HOST_GRAPHICS_TYPE);

        // xforms creation
        xformsBuffers[i] = onyx_request_buffer_region(
            memory, sizeof(Xforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            ONYX_MEMORY_HOST_GRAPHICS_TYPE);

        // lights creation
        lightsBuffers[i] = onyx_request_buffer_region(
            memory, sizeof(Lights), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            ONYX_MEMORY_HOST_GRAPHICS_TYPE);

        materialsBuffers[i] = onyx_request_buffer_region(
            memory, sizeof(Material) * ONYX_S_MAX_MATERIALS,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, ONYX_MEMORY_HOST_GRAPHICS_TYPE);

        VkDescriptorBufferInfo camInfo = {.buffer = cameraBuffers[i].buffer,
                                          .offset = cameraBuffers[i].offset,
                                          .range  = cameraBuffers[i].size};

        VkDescriptorBufferInfo xformInfo = {.buffer = xformsBuffers[i].buffer,
                                            .offset = xformsBuffers[i].offset,
                                            .range  = xformsBuffers[i].size};

        VkDescriptorBufferInfo lightInfo = {.buffer = lightsBuffers[i].buffer,
                                            .offset = lightsBuffers[i].offset,
                                            .range  = lightsBuffers[i].size};

        VkDescriptorBufferInfo materialInfo = {
            .buffer = materialsBuffers[i].buffer,
            .offset = materialsBuffers[i].offset,
            .range  = materialsBuffers[i].size};

        VkWriteDescriptorSet writes[] = {
            {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
             .dstArrayElement = 0,
             .dstSet          = descriptorSets[i][DESC_SET_MAIN],
             .dstBinding      = 0,
             .descriptorCount = 1,
             .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
             .pBufferInfo     = &camInfo},
            {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
             .dstArrayElement = 0,
             .dstSet          = descriptorSets[i][DESC_SET_MAIN],
             .dstBinding      = 1,
             .descriptorCount = 1,
             .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
             .pBufferInfo     = &xformInfo},
            {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
             .dstArrayElement = 0,
             .dstSet          = descriptorSets[i][DESC_SET_MAIN],
             .dstBinding      = 2,
             .descriptorCount = 1,
             .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
             .pBufferInfo     = &lightInfo},
            {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
             .dstArrayElement = 0,
             .dstSet          = descriptorSets[i][DESC_SET_MAIN],
             .dstBinding      = 4,
             .descriptorCount = 1,
             .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
             .pBufferInfo     = &materialInfo}};

        vkUpdateDescriptorSets(device, LEN(writes), writes, 0, NULL);
    }

    updateGbufferDescriptors();
}

static void
updateTexture(const uint32_t frameIndex, const OnyxImage* img,
              const uint32_t texId)
{
    VkDescriptorImageInfo textureInfo = {
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .imageView   = img->view,
        .sampler     = img->sampler};

    VkWriteDescriptorSet write = {
        .sType      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext      = NULL,
        .dstSet     = descriptorSets[frameIndex][DESC_SET_MAIN],
        .dstBinding = 3,
        .dstArrayElement = texId,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo      = &textureInfo};

    vkUpdateDescriptorSets(device, 1, &write, 0, NULL);

    printf("Updated Texture %d frame %d\n", texId, frameIndex);
}

static void
generateGBuffer(VkCommandBuffer cmdBuf, const OnyxScene* scene,
                const uint32_t frameIndex, uint32_t frame_width,
                uint32_t frame_height)
{
    VkClearValue clearValueColor = {0.1f, 0.1f, 0.1f, 1.0f};
    VkClearValue clearValueMatid = {0};
    VkClearValue clearValueDepth = {1.0, 0};

    VkClearValue clears[] = {clearValueColor, clearValueColor, clearValueColor,
                             clearValueMatid, clearValueDepth};

    VkRenderPassBeginInfo rpassInfo = {
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .clearValueCount = LEN(clears),
        .pClearValues    = clears,
        .renderArea      = {{0, 0}, {frame_width, frame_height}},
        .renderPass      = gbufferRenderPass,
        .framebuffer     = gframebuffer};

    vkCmdBeginRenderPass(cmdBuf, &rpassInfo, VK_SUBPASS_CONTENTS_INLINE);

    for (int pipeId = 0; pipeId < GBUFFER_PIPELINE_COUNT; pipeId++)
    {
        vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          gbufferPipelines[pipeId]);
        uint32_t                    primCount;
        const OnyxPrimitiveHandle* prim_handles =
            onyx_get_primlist_prims(&pipelinePrimLists[pipeId], &primCount);
        for (int i = 0; i < primCount; i++)
        {
            OnyxPrimitiveHandle  prim_handle = prim_handles[i];
            const OnyxPrimitive* prim =
                onyx_scene_get_primitive_const(scene, prim_handle);
            OnyxMaterialHandle matId = prim->material;
            OnyxXform          xform = prim->xform;
            vkCmdPushConstants(cmdBuf, pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4),
                               xform.e);
            vkCmdPushConstants(cmdBuf, pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, sizeof(Mat4),
                               sizeof(uint32_t), &prim_handle.id);
            vkCmdPushConstants(
                cmdBuf, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                sizeof(Mat4) + sizeof(uint32_t), sizeof(uint32_t), &matId);
            onyx_draw_geo(cmdBuf, prim->geo, 1);
        }
    }

    vkCmdEndRenderPass(cmdBuf);
}

static void
shadowPass(VkCommandBuffer cmdBuf, const uint32_t frameIndex,
           uint32_t windowWidth, uint32_t windowHeight)
{
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                      raytracePipeline);

    vkCmdTraceRaysKHR(
        cmdBuf, &shaderBindingTable.raygen_table, &shaderBindingTable.miss_table,
        &shaderBindingTable.hit_table, &shaderBindingTable.callable_table,
        windowWidth, windowHeight, 1);
}

static void
deferredRender(VkCommandBuffer cmdBuf, const uint32_t frameIndex,
               uint32_t windowWidth, uint32_t windowHeight)
{
    VkClearValue clearValueColor = {0.1f, 0.1f, 0.1f, 1.0f};

    VkRenderPassBeginInfo rpassInfo = {
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .clearValueCount = 1,
        .pClearValues    = &clearValueColor,
        .renderArea      = {{0, 0}, {windowWidth, windowHeight}},
        .renderPass      = deferredRenderPass,
        .framebuffer     = swapImageBuffer[frameIndex]};

    vkCmdBeginRenderPass(cmdBuf, &rpassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      defferedPipeline);

    vkCmdDraw(cmdBuf, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmdBuf);
}

static void
sortPipelinePrims(const OnyxScene* scene)
{
    for (int i = 0; i < GBUFFER_PIPELINE_COUNT; i++)
    {
        onyx_clear_prim_list(&pipelinePrimLists[i]);
    }
    obint                       prim_count = 0;
    const OnyxPrimitiveHandle* prims =
        onyx_scene_get_dirty_primitives(scene, &prim_count);
    for (obint primId = 0; primId < prim_count; primId++)
    {
        AttrMask                   attrMask = 0;
        const OnyxPrimitiveHandle handle   = prims[primId];
        const OnyxPrimitive* prim = onyx_scene_get_primitive_const(scene, handle);
        if (prim->flags & ONYX_PRIM_INVISIBLE_BIT)
            continue;
        const OnyxGeometry* geo = prim->geo;
        for (int i = 0; i < geo->templ.attribute_count; i++)
        {
            OnyxAttributeType name = geo->templ.attribute_types[i];
            if (name == ONYX_ATTRIBUTE_TYPE_POS)
                attrMask |= POS_BIT;
            if (name == ONYX_ATTRIBUTE_TYPE_NORMAL)
                attrMask |= NORMAL_BIT;
            if (name == ONYX_ATTRIBUTE_TYPE_UV)
                attrMask |= UV_BIT;
            if (name == ONYX_ATTRIBUTE_TYPE_TANGENT)
                attrMask |= TAN_BIT;
            if (name == ONYX_ATTRIBUTE_TYPE_SIGN)
                attrMask |= SIGN_BIT;
        }
        if (attrMask == POS_NOR_UV_TAN_MASK)
            onyx_add_prim_to_list(
                handle, &pipelinePrimLists[PIPELINE_GBUFFER_POS_NOR_UV_TAN]);
        else if (attrMask == POS_NOR_UV_MASK)
            onyx_add_prim_to_list(handle,
                               &pipelinePrimLists[PIPELINE_GBUFFER_POS_NOR_UV]);
        else if (attrMask == POS_MASK)
            onyx_add_prim_to_list(handle,
                               &pipelinePrimLists[PIPELINE_GBUFFER_POS]);
        else
        {
            printf("Attributes not supported!\n");
            assert(0);
        }
        // if (mat->textureAlbedo && mat->textureRoughness &&
        // mat->textureNormal)
        //     addPrimToPipelinePrimList(primId,
        //     &pipelinePrimLists[PIPELINE_TAN]);
        // else if (mat->textureAlbedo && mat->textureRoughness)
        //     addPrimToPipelinePrimList(primId,
        //     &pipelinePrimLists[PIPELINE_REG]);
        // else if (mat->textureAlbedo + mat->textureNormal + mat->textureNormal
        // == 0 )
        //     addPrimToPipelinePrimList(primId,
        //     &pipelinePrimLists[PIPELINE_NO_MAPS]);
        // else
        //     assert(0 && "currently prims must have albedo and roughness
        //     textures");
    }
}

static void
updateRenderCommands(VkCommandBuffer cmdBuf, const OnyxScene* scene,
                     const WoadFrame* frame, uint32_t region_x,
                     uint32_t region_y, uint32_t region_width,
                     uint32_t region_height)
{
    uint32_t frameIndex = frame->index;
    onyx_cmd_set_viewport_scissor(cmdBuf, region_x, region_y, region_width,
                               region_height);

    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 0, 2,
                            descriptorSets[frameIndex], 0, NULL);

    uint32_t light_count = onyx_scene_get_light_count(scene);
    vkCmdPushConstants(
        cmdBuf, pipelineLayout,
        VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        sizeof(Mat4) + sizeof(uint32_t) * 2, sizeof(uint32_t), &light_count);

    // ensures that previous frame has already read gbuffer, by ensuring that
    // all previous commands have completed fragment shader reads.
    // we could potentially be more fine-grained by using a VkEvent
    // to wait specifically for that exact read to happen.
    onyx_v_MemoryBarrier(cmdBuf, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_DEPENDENCY_BY_REGION_BIT, VK_ACCESS_SHADER_READ_BIT,
                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

    generateGBuffer(cmdBuf, scene, frameIndex, frame->width, frame->height);

    if (raytracing_disabled)
    {
    }
    else
    {
        onyx_v_MemoryBarrier(
            cmdBuf, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        vkCmdBindDescriptorSets(
            cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout, 0,
            2, descriptorSets[frameIndex], 0, NULL);

        shadowPass(cmdBuf, frameIndex, region_width, region_height);

        onyx_v_MemoryBarrier(
            cmdBuf, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }

    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 0, 2,
                            descriptorSets[frameIndex], 0, NULL);

    // viewport.height = windowHeight;
    // viewport.y = 0;

    // vkCmdSetViewport(cmdBuf, 0, 1, &viewport);

    deferredRender(cmdBuf, frameIndex, frame->width, frame->height);
}

static void
freeImages(void)
{
    onyx_free_image(&renderTargetDepth);
    onyx_free_image(&imageWorldP);
    onyx_free_image(&imageNormal);
    onyx_free_image(&imageShadow);
    onyx_free_image(&imageRoughness);
    onyx_free_image(&imageAlbedo);
}

static void
onDirtyFrame(const WoadFrame *fb)
{
    static uint32_t last_width = 0, last_height = 0;

    onyx_destroy_framebuffer(device, swapImageBuffer[fb->index]);
    initSwapFramebuffer(fb);

    if (fb->width != last_width || fb->height != last_height)
    {
        vkDeviceWaitIdle(device);
        freeImages();
        initAttachments(fb->width, fb->height);
        onyx_destroy_framebuffer(device, gframebuffer);
        initGbufferFramebuffer(fb->width, fb->height);
        updateGbufferDescriptors();
    }
}

static void
updateCamera(const OnyxScene* scene, uint32_t index)
{
    Camera* uboCam = (Camera*)cameraBuffers[index].host_data;
    uboCam->view   = onyx_scene_get_camera_view(scene);
    uboCam->proj   = onyx_scene_get_camera_projection(scene);
    // printf("Proj:\n");
    // coal_PrintMat4(&proj);
    // printf("View:\n");
    // coal_PrintMat4(&view);
    uboCam->camera = onyx_scene_get_camera_xform(scene);
}

static void
updateFastXforms(uint32_t frameIndex, uint32_t primIndex)
{
    assert(primIndex < 16);
    Xforms* xforms = (Xforms*)xformsBuffers[frameIndex].host_data;
    // coal_Copy_Mat4(scene->xforms[primIndex], xforms->xform[primIndex].x);
}

static void
updateLight(const OnyxScene* scene, uint32_t frameIndex, uint32_t lightIndex)
{
}

static void
updateMaterials(const OnyxScene* scene, uint32_t frameIndex)
{
    _Static_assert(sizeof(OnyxMaterial) == 4 * 8,
                   "Check shader material against OnyxMaterial\n");
    obint                matcount  = 0;
    const OnyxMaterial* materials = onyx_scene_get_materials(scene, &matcount);
    memcpy(materialsBuffers[frameIndex].host_data, materials,
           sizeof(Material) * matcount);
}

// possibly because transforms changed, we only need to rebuild the tlas
static void
buildTlas(const OnyxScene* scene, int frame_index)
{
    HellArray xforms;

    hell_create_array_old(8, sizeof(CoalMat4), NULL, NULL, &xforms);

    obint                  prim_count = 0;
    const OnyxPrimitive*  prims = onyx_scene_get_primitives(scene, &prim_count);

    if (tlas[frame_index].buffer_region.size != 0)
        onyx_destroy_acceleration_struct(device, &tlas[frame_index]);

    if (prim_count > 0)
    {
        for (int i = 0; i < prim_count; i++)
        {
            // note that we are assuming here that number of visible prims ==
            // length of the blas array
            if (prims[i].flags & ONYX_PRIM_INVISIBLE_BIT)
                continue;
            hell_array_push(&xforms, &prims[i].xform);
        }
        onyx_build_tlas(memory, blas_array.count, blas_array.elems, xforms.elems,
                       &tlas[frame_index]);
    }

    hell_destroy_array(&xforms, NULL);
}

static void
buildAccelerationStructures(const OnyxScene* scene)
{
    obint                  prim_count = 0;
    const OnyxPrimitive*  prims = onyx_scene_get_primitives(scene, &prim_count);

    for (int i = 0; i < blas_array.count; i++)
    {
        AccelerationStructure *blas = &blas_array.elems[i];
        onyx_destroy_acceleration_struct(device, blas);
    }

    accel_struct_arr_set_count(&blas_array, 0);

    if (prim_count > 0)
    {
        for (int i = 0; i < prim_count; i++)
        {
            if (prims[i].flags & ONYX_PRIM_INVISIBLE_BIT)
                continue;
            AccelerationStructure blas = {};
            onyx_build_blas(memory, prims[i].geo, &blas);
            accel_struct_arr_push(&blas_array, blas);
        }
    }

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        buildTlas(scene, i);
    }

    printf(">>>>> Built acceleration structures\n");
}

void
woad_Render(const OnyxScene* scene, const WoadFrame* fb, uint32_t x,
            uint32_t y, uint32_t width, uint32_t height, VkCommandBuffer cmdbuf)
{
    // assert(x + width  <= fb->width);
    // assert(y + height <= fb->height);
    static uint8_t cameraNeedUpdate    = MAX_FRAMES_IN_FLIGHT;
    static uint8_t asNeedUpdate        = MAX_FRAMES_IN_FLIGHT;
    // static uint8_t xformsNeedUpdate    = MAX_FRAMES_IN_FLIGHT;
    static uint8_t lightsNeedUpdate    = MAX_FRAMES_IN_FLIGHT;
    static uint8_t texturesNeedUpdate  = MAX_FRAMES_IN_FLIGHT;
    static uint8_t materialsNeedUpdate = MAX_FRAMES_IN_FLIGHT;

    int                  frameIndex = fb->index;
    OnyxSceneDirtyFlags scene_dirt = onyx_scene_get_dirt(scene);
    if (scene_dirt)
    {
        if (scene_dirt & ONYX_SCENE_CAMERA_VIEW_BIT)
            cameraNeedUpdate = MAX_FRAMES_IN_FLIGHT;
        if (scene_dirt & ONYX_SCENE_CAMERA_PROJ_BIT)
            cameraNeedUpdate = MAX_FRAMES_IN_FLIGHT;
        if (scene_dirt & ONYX_SCENE_LIGHTS_BIT)
        {
            lightsNeedUpdate = MAX_FRAMES_IN_FLIGHT;
        }
        if (scene_dirt & ONYX_SCENE_MATERIALS_BIT)
            materialsNeedUpdate = MAX_FRAMES_IN_FLIGHT;
        if (scene_dirt & ONYX_SCENE_TEXTURES_BIT)
        {
            texturesNeedUpdate = MAX_FRAMES_IN_FLIGHT;
        }
        if (scene_dirt & ONYX_SCENE_PRIMS_BIT)
        {
            printf("WOAD: PRIMS DIRTY\n");
            sortPipelinePrims(scene);
            if (!raytracing_disabled)
            {
                buildAccelerationStructures(scene);
                asNeedUpdate = MAX_FRAMES_IN_FLIGHT;
            }
        }
        else if (scene_dirt & ONYX_SCENE_XFORMS_BIT)
        {
            if (!raytracing_disabled)
            {
                asNeedUpdate = MAX_FRAMES_IN_FLIGHT;
            }
        }
    }
    if (fb->dirty)
    {
        onDirtyFrame(fb);
    }

    if (asNeedUpdate)
    {
        buildTlas(scene, frameIndex);
        updateASDescriptors(frameIndex);
        asNeedUpdate--;
    }
    if (cameraNeedUpdate)
    {
        updateCamera(scene, frameIndex);
        cameraNeedUpdate--;
    }
    if (lightsNeedUpdate)
    {
        obint       light_count;
        OnyxLight* scene_lights = onyx_scene_get_lights(scene, &light_count);
        Lights*     lights       = (Lights*)lightsBuffers[frameIndex].host_data;
        memcpy(lights, scene_lights, sizeof(Light) * light_count);
        lightsNeedUpdate--;
        printf("Tanto: lights sync\n");
        onyx_print_light_info(scene);
    }
    if (materialsNeedUpdate)
    {
        updateMaterials(scene, frameIndex);
        materialsNeedUpdate--;
    }
    if (texturesNeedUpdate) // TODO update all tex
    {
        printf("texturesNeedUpdate %d\n", texturesNeedUpdate);
        obint               tex_count = 0;
        const OnyxTexture* tex = onyx_scene_get_textures(scene, &tex_count);
        // remember, 1 is the first valid texture index
        for (int i = 0; i < tex_count; i++)
        {
            updateTexture(frameIndex, tex[i].dev_image, i);
        }
        texturesNeedUpdate--;
    }

    updateRenderCommands(cmdbuf, scene, fb, x, y, width, height);
}

void
woad_Init(const OnyxInstance* instance_, OnyxMemory* memory_,
          VkImageLayout finalColorLayout, VkImageLayout finalDepthLayout,
          const OnyxSwapchain *swapchain,
          Woad_Settings_Flags flags)
{
    hell_print("Creating Woad renderer...\n");
    instance = instance_;
    if (flags & WOAD_SETTINGS_NO_RAYTRACE_BIT)
        raytracing_disabled = true;
    for (int i = 0; i < GBUFFER_PIPELINE_COUNT; i++)
    {
        pipelinePrimLists[i] = onyx_create_prim_list(8);
    }

    blas_array = accel_struct_arr_create(NULL);

    device = onyx_get_device(instance);
    graphic_queue_family_index =
        onyx_queue_family_index(instance, ONYX_QUEUE_GRAPHICS_TYPE);
    memory = memory_;

    uint32_t width, height;
    VkFormat format;
    width = onyx_get_swapchain_width(swapchain);
    height = onyx_get_swapchain_height(swapchain);
    format = onyx_get_swapchain_format(swapchain);


    initAttachments(width, height);
    hell_print(">> Woad: attachments initialized. \n");
    initGbufRenderPass();
    onyx_create_render_pass_color(device, VK_IMAGE_LAYOUT_UNDEFINED,
                                finalColorLayout, VK_ATTACHMENT_LOAD_OP_CLEAR,
                                format, &deferredRenderPass);
    hell_print(">> Woad: renderpasses initialized. \n");
    initGbufferFramebuffer(width, height);
    for (int i = 0; i < 2; i++)
    {
        WoadFrame f = {
            .dirty = true,
            .format = format,
            .width = width,
            .height = height,
            .index = i,
            .view = onyx_get_swapchain_image_view(swapchain, i),
        };
        initSwapFramebuffer(&f);
    }
    hell_print(">> Woad: framebuffers initialized. \n");
    initDescriptorSetsAndPipelineLayouts();
    hell_print(">> Woad: descriptor sets and pipeline layouts initialized. \n");
    updateDescriptors();
    hell_print(">> Woad: descriptors updated. \n");
    initPipelines(false);
    hell_print(">> Woad: pipelines initialized. \n");
    hell_print(">> Woad: initialization complete. \n");
}

void
woad_Cleanup(void)
{
    for (int i = 0; i < GBUFFER_PIPELINE_COUNT; i++)
    {
        vkDestroyPipeline(device, gbufferPipelines[i], NULL);
    }
    vkDestroyPipeline(device, defferedPipeline, NULL);
    vkDestroyPipeline(device, raytracePipeline, NULL);
    AccelerationStructure* blasses = blas_array.elems;
    for (int i = 0; i < blas_array.count; i++)
    {
        AccelerationStructure* blas = &blasses[i];
        if (blas->buffer_region.size != 0)
            onyx_destroy_acceleration_struct(device, blas);
    }
    onyx_destroy_shader_binding_table(&shaderBindingTable);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (tlas[i].buffer_region.size != 0)
            onyx_destroy_acceleration_struct(device, &tlas[i]);
        vkDestroyDescriptorPool(device, descriptorPool, NULL);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayouts[i], NULL);
        onyx_free_buffer(&cameraBuffers[i]);
        onyx_free_buffer(&xformsBuffers[i]);
        onyx_free_buffer(&lightsBuffers[i]);
        onyx_free_buffer(&materialsBuffers[i]);
    }
    vkDestroyRenderPass(device, gbufferRenderPass, NULL);
    vkDestroyRenderPass(device, deferredRenderPass, NULL);
    vkDestroyPipelineLayout(device, pipelineLayout, NULL);
}

WoadFrame
woad_Frame(const OnyxSwapchainImage *img)
{
    // track uuid for each swapimage
    static int64_t last_img_uuids[2] = {-1, -1};

    const uint32_t idx = img->index;
    int64_t cur_uuid = img->swapchain->image_uuid[idx];

    bool dirty = last_img_uuids[idx] != cur_uuid;
    last_img_uuids[idx] = cur_uuid;

    WoadFrame f = {
        .dirty = dirty,
        .format = onyx_get_swapchain_format(img->swapchain),
        .width = onyx_get_swapchain_width(img->swapchain),
        .height = onyx_get_swapchain_height(img->swapchain),
        .index = idx,
        .view = onyx_get_swapchain_image_view(img->swapchain, idx),
    };

    return f;
}
