#include "coal/m_math.h"
#include "coal/util.h"
#include "obsidian/r_geo.h"
#include "obsidian/s_scene.h"
#include "obsidian/v_image.h"
#include "obsidian/v_memory.h"
#include "obsidian/v_vulkan.h"
#include <memory.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <obsidian/r_render.h>
#include <obsidian/v_video.h>
#include <obsidian/v_swapchain.h>
#include <obsidian/t_def.h>
#include <obsidian/u_ui.h>
#include <obsidian/t_utils.h>
#include <obsidian/r_pipeline.h>
#include <obsidian/r_raytrace.h>
#include <obsidian/r_renderpass.h>
#include <obsidian/v_command.h>
#include <obsidian/r_api.h>
#include <vulkan/vulkan_core.h>
#include <obsidian/v_private.h>

#define SPVDIR "/home/michaelb/dev/tanto/shaders/spv"

typedef Obdn_V_Command               Command;
typedef Obdn_V_Image                 Image;
typedef Obdn_S_Light                 Light;
typedef Obdn_S_Material              Material;
typedef Obdn_V_BufferRegion          BufferRegion;
typedef Obdn_R_AccelerationStructure AccelerationStructure;
typedef Obdn_R_Primitive             Prim;

enum {
    PIPE_LAYOUT_MAIN,
};

enum {
    DESC_SET_MAIN,
    DESC_SET_DEFERRED,
    DESC_SET_COUNT
};

enum {
    PIPELINE_GBUFFER_NO_TAN,
    PIPELINE_GBUFFER_TAN,
    GBUFFER_PIPELINE_COUNT
};

_Static_assert(GBUFFER_PIPELINE_COUNT < OBDN_MAX_PIPELINES, "GRAPHICS_PIPELINE_COUNT must be less than OBDN_MAX_PIPELINES");

#define MAX_PRIM_COUNT OBDN_S_MAX_PRIMS

typedef struct {
    Light light[OBDN_S_MAX_LIGHTS];
} Lights;

typedef struct {
    Mat4 xform[MAX_PRIM_COUNT];
} Xforms;

typedef struct {
    Mat4 view;
    Mat4 proj;
    Mat4 camera;
} Camera;

#define MAX_FRAMES_IN_FLIGHT 2

static VkRenderPass  renderpass;
static VkRenderPass  gbufferRenderPass;
static VkRenderPass  deferredRenderPass;

static VkFramebuffer framebuffers[MAX_FRAMES_IN_FLIGHT];
static VkFramebuffer gbuffers[MAX_FRAMES_IN_FLIGHT];
static VkFramebuffer swapImageBuffer[MAX_FRAMES_IN_FLIGHT];

static VkPipeline                gbufferPipelines[GBUFFER_PIPELINE_COUNT];
static VkPipeline                defferedPipeline;

static VkPipeline                raytracePipeline;
static Obdn_R_ShaderBindingTable shaderBindingTable;

static BufferRegion cameraBuffers[MAX_FRAMES_IN_FLIGHT];
static BufferRegion xformsBuffers[MAX_FRAMES_IN_FLIGHT];
static BufferRegion lightsBuffers[MAX_FRAMES_IN_FLIGHT];
static BufferRegion materialsBuffers[MAX_FRAMES_IN_FLIGHT];

static const Obdn_S_Scene* scene;

static Obdn_S_PrimitiveList pipelinePrimLists[GBUFFER_PIPELINE_COUNT];

// raytrace stuff

static AccelerationStructure blasses[OBDN_S_MAX_PRIMS];
static AccelerationStructure tlas;

// raytrace stuff

static Command renderCommands[MAX_FRAMES_IN_FLIGHT];

static VkDescriptorSetLayout descriptorSetLayouts[DESC_SET_COUNT];
static Obdn_R_Description    descriptions[MAX_FRAMES_IN_FLIGHT];

static VkPipelineLayout pipelineLayout;

static uint32_t windowWidth;
static uint32_t windowHeight;

static Image renderTargetDepth;
static Image imageWorldP;
static Image imageNormal;
static Image imageShadow;
static Image imageAlbedo;
static Image imageRoughness;

_Static_assert(OBDN_S_MAX_LIGHTS <= 16, "must be less than or equal to number of bits in our shadow mask");
static const VkFormat formatImageP         = VK_FORMAT_R32G32B32A32_SFLOAT;
static const VkFormat formatImageN         = VK_FORMAT_R32G32B32A32_SFLOAT;
static const VkFormat formatImageShadow    = VK_FORMAT_R16_UINT; // maximum of 16 lights.
static const VkFormat formatImageAlbedo    = VK_FORMAT_R8G8B8A8_UNORM;
static const VkFormat formatImageRoughness = VK_FORMAT_R16_UNORM;


// declarations for overview and navigation
static void initAttachments(void);
static void initRenderPass(void);
static void initFramebuffers(void);
static void initDescriptorSetsAndPipelineLayouts(void);
static void initPipelines(void);
static void updateDescriptors(void);
static void updateRenderCommands(const uint32_t frameIndex);
static void onSwapchainRecreate(void);
static void updateLight(uint32_t frameIndex, uint32_t lightIndex);
static void updateCamera(uint32_t index);
static void updateXform(uint32_t frameIndex, uint32_t primIndex);
static void syncScene(const uint32_t frameIndex);

static Obdn_R_Import ri;

void        r_InitRenderer(const Obdn_S_Scene* scene);
VkSemaphore r_Render(uint32_t frameIndex, VkSemaphore waitSemephore); 
void        r_CleanUp(void);
uint8_t     r_GetMaxFramesInFlight(void) { return MAX_FRAMES_IN_FLIGHT; }


Obdn_R_Export handshake(Obdn_R_Import rimport)
{
    ri = rimport;

    Obdn_R_Export export = {0};
    export.init = r_InitRenderer;
    export.cleanUp = r_CleanUp;
    export.render = r_Render;
    export.getMaxFramesInFlight = r_GetMaxFramesInFlight;

    return export;
}

static void initAttachments(void)
{
    renderTargetDepth = obdn_v_CreateImage(
        windowWidth, windowHeight, obdn_r_GetDepthFormat(),
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT, VK_SAMPLE_COUNT_1_BIT, 1,
        OBDN_V_MEMORY_DEVICE_TYPE);

    imageWorldP = obdn_v_CreateImage(
        windowWidth, windowHeight, formatImageP,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, VK_SAMPLE_COUNT_1_BIT, 1,
        OBDN_V_MEMORY_DEVICE_TYPE);

    imageNormal = obdn_v_CreateImage(
        windowWidth, windowHeight, formatImageN,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, VK_SAMPLE_COUNT_1_BIT, 1,
        OBDN_V_MEMORY_DEVICE_TYPE);

    imageShadow = obdn_v_CreateImage(
        windowWidth, windowHeight, formatImageShadow,
        VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, VK_SAMPLE_COUNT_1_BIT, 1,
        OBDN_V_MEMORY_DEVICE_TYPE);

    imageAlbedo = obdn_v_CreateImage(
        windowWidth, windowHeight, formatImageAlbedo,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, VK_SAMPLE_COUNT_1_BIT, 1,
        OBDN_V_MEMORY_DEVICE_TYPE);
    
    imageRoughness = obdn_v_CreateImage(
        windowWidth, windowHeight, formatImageRoughness,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, VK_SAMPLE_COUNT_1_BIT, 1,
        OBDN_V_MEMORY_DEVICE_TYPE);

    obdn_v_TransitionImageLayout(imageShadow.layout, VK_IMAGE_LAYOUT_GENERAL, &imageShadow);
}

static void initRenderPass(void)
{
    obdn_r_CreateRenderPass_ColorDepth(
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
        VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE,
        obdn_v_GetSwapFormat(), obdn_r_GetDepthFormat(), &renderpass);
    printf("Created renderpass 1...\n");

    // gbuffer renderpass
    {
        VkAttachmentDescription attachmentWorldP = {
            .flags = 0,
            .format = formatImageP,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_GENERAL };

        VkAttachmentDescription attachmentNormal = {
            .flags = 0,
            .format = formatImageN,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_GENERAL };

        VkAttachmentDescription attachmentAlbedo = {
            .flags = 0,
            .format = formatImageAlbedo,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_GENERAL };

        VkAttachmentDescription attachmentRoughness = {
            .flags = 0,
            .format = formatImageRoughness,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_GENERAL };

        VkAttachmentDescription attachmentDepth = {
            .flags = 0,
            .format = obdn_r_GetDepthFormat(),
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };


        VkAttachmentReference refWorldP = {
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

        VkAttachmentReference refNormal = {
            .attachment = 1,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

        VkAttachmentReference refAlbedo = {
            .attachment = 2,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

        VkAttachmentReference refRough = {
            .attachment = 3,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

        VkAttachmentReference refDepth = {
            .attachment = 4,
            .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkAttachmentReference colorRefs[] = {refWorldP, refNormal, refAlbedo, refRough};

        VkSubpassDescription subpass = {
            .flags = 0,
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .inputAttachmentCount = 0,
            .pInputAttachments = NULL,
            .colorAttachmentCount = OBDN_ARRAY_SIZE(colorRefs),
            .pColorAttachments = colorRefs,
            .pResolveAttachments = NULL,
            .pDepthStencilAttachment = &refDepth,
            .preserveAttachmentCount = 0,
            .pPreserveAttachments = NULL};

        VkSubpassDependency dep1 = {
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
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
            .dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = 0,
            .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT};

        VkSubpassDependency deps[] = {dep1, dep2};

        VkAttachmentDescription attachments[] = {
            attachmentWorldP, attachmentNormal, attachmentAlbedo, attachmentRoughness, attachmentDepth};

        VkRenderPassCreateInfo rpiInfo = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .attachmentCount = OBDN_ARRAY_SIZE(attachments),
            .pAttachments = attachments,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = OBDN_ARRAY_SIZE(deps),
            .pDependencies = deps};

        V_ASSERT(
            vkCreateRenderPass(device, &rpiInfo, NULL, &gbufferRenderPass));
    }
    printf("Created renderpass 2...\n");

    obdn_r_CreateRenderPass_Color(VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                  VK_ATTACHMENT_LOAD_OP_CLEAR,
                                  obdn_v_GetSwapFormat(), &deferredRenderPass);
}

static void initFramebuffers(void)
{
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
    {
        {
            const Obdn_R_Frame* frame = obdn_v_GetFrame(i);
            const VkImageView attachments[] = {
                frame->view, renderTargetDepth.view
            };

            const VkFramebufferCreateInfo fbi = {
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .pNext = NULL,
                .flags = 0,
                .renderPass = renderpass,
                .attachmentCount = 2,
                .pAttachments = attachments,
                .width = windowWidth,
                .height = windowHeight,
                .layers = 1,
            };

            V_ASSERT( vkCreateFramebuffer(device, &fbi, NULL, &framebuffers[i]) );
        }
        {
            const VkImageView attachments[] = {imageWorldP.view, imageNormal.view, imageAlbedo.view, imageRoughness.view, renderTargetDepth.view};

            const VkFramebufferCreateInfo fbi = {
                .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .pNext           = NULL,
                .flags           = 0,
                .renderPass      = gbufferRenderPass,
                .attachmentCount = 5,
                .pAttachments    = attachments,
                .width           = windowWidth,
                .height          = windowHeight,
                .layers          = 1
            };

            V_ASSERT( vkCreateFramebuffer(device, &fbi, NULL, &gbuffers[i]));
        }
        {
            const Obdn_R_Frame* frame = obdn_v_GetFrame(i);

            const VkFramebufferCreateInfo fbi = {
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .pNext = NULL,
                .flags = 0,
                .renderPass = deferredRenderPass,
                .attachmentCount = 1,
                .pAttachments = &frame->view,
                .width = windowWidth,
                .height = windowHeight,
                .layers = 1,
            };

            V_ASSERT( vkCreateFramebuffer(device, &fbi, NULL, &swapImageBuffer[i]) );
        }
    }
}

static void initDescriptorSetsAndPipelineLayouts(void)
{
    const Obdn_R_DescriptorSetInfo descriptorSets[] = {{
        .bindingCount = 5,
        .bindings = {{ 
            // camera
            .descriptorCount = 1,
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
        },{ // xforms
            .descriptorCount = 1,
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT
        },{ // lights
            .descriptorCount = 1,
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR
        },{ // textures
            .descriptorCount = OBDN_S_MAX_TEXTURES, // because this is an array of samplers. others are structs of arrays.
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .bindingFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT
        },{ // materials
            .descriptorCount = 1, // because this is an array of samplers. others are structs of arrays.
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        }}
    },{
        .bindingCount = 6,
        .bindings = {{ 
            // worldp storage image
            .descriptorCount = 1,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        },{ // normal storage image
            .descriptorCount = 1,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        },{ // albedo storage image
            .descriptorCount = 1,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
        },{ // shadow storage image
            .descriptorCount = 1,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        },{ // roughness storage image
            .descriptorCount = 1,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT 
        },{ // top level AS
            .descriptorCount = 1,
            .type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        }}
    }};

    obdn_r_CreateDescriptionsAndLayouts(OBDN_ARRAY_SIZE(descriptorSets),
                                        descriptorSets, descriptorSetLayouts,
                                        MAX_FRAMES_IN_FLIGHT, descriptions);

    const VkPushConstantRange pcPrimId = {
        .offset = 0,
        .size = sizeof(uint32_t) * 2, //prim id, material id
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT
    };

    // light count
    const VkPushConstantRange pcFrag = {
        .offset = sizeof(uint32_t) * 2,
        .size = sizeof(uint32_t),
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR
    };

    const VkPushConstantRange ranges[] = {pcPrimId, pcFrag};

    const Obdn_R_PipelineLayoutInfo pipeLayoutInfos[] = {{
        .descriptorSetCount = OBDN_ARRAY_SIZE(descriptorSets), 
        .descriptorSetLayouts = descriptorSetLayouts,
        .pushConstantCount = OBDN_ARRAY_SIZE(ranges),
        .pushConstantsRanges = ranges
    }};

    obdn_r_CreatePipelineLayouts(1, pipeLayoutInfos, &pipelineLayout);
}

static void initPipelines(void)
{
    const Obdn_R_AttributeSize regularPrimAttrSizes[3] = {12, 12, 8};
    const Obdn_R_AttributeSize tangetPrimAttrSizes[4]  = {12, 12, 8, 12};

    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

    const Obdn_R_GraphicsPipelineInfo gPipelineInfos[] = {{
        .renderPass = gbufferRenderPass, 
        .layout     = pipelineLayout,
        .sampleCount = VK_SAMPLE_COUNT_1_BIT,
        .frontFace   = VK_FRONT_FACE_CLOCKWISE,
        .attachmentCount = 4,
        .vertexDescription = obdn_r_GetVertexDescription(3, regularPrimAttrSizes),
        .dynamicStateCount = OBDN_ARRAY_SIZE(dynamicStates),
        .pDynamicStates = dynamicStates,
        .vertShader = SPVDIR"/regular-vert.spv",
        .fragShader = SPVDIR"/gbuffer-frag.spv",
    },{
        .renderPass = gbufferRenderPass, 
        .layout     = pipelineLayout,
        .sampleCount = VK_SAMPLE_COUNT_1_BIT,
        .frontFace   = VK_FRONT_FACE_CLOCKWISE,
        .attachmentCount = 4,
        .dynamicStateCount = OBDN_ARRAY_SIZE(dynamicStates),
        .pDynamicStates = dynamicStates,
        .vertexDescription = obdn_r_GetVertexDescription(4, tangetPrimAttrSizes),
        .vertShader = SPVDIR"/tangent-vert.spv",
        .fragShader = SPVDIR"/gbuffertan-frag.spv"
    }};

    const Obdn_R_GraphicsPipelineInfo defferedPipeInfo = {
        .renderPass = deferredRenderPass, 
        .layout     = pipelineLayout,
        .sampleCount = VK_SAMPLE_COUNT_1_BIT,
        .frontFace   = VK_FRONT_FACE_CLOCKWISE,
        .dynamicStateCount = OBDN_ARRAY_SIZE(dynamicStates),
        .pDynamicStates = dynamicStates,
        .vertShader = obdn_r_FullscreenTriVertShader(),
        .fragShader = SPVDIR"/deferred-frag.spv"
    };

    const Obdn_R_RayTracePipelineInfo rtPipelineInfo = {
        .layout = pipelineLayout,
        .raygenCount = 1,
        .raygenShaders = (char*[]){SPVDIR"/shadow-rgen.spv"},
        .missCount = 1,
        .missShaders = (char*[]){SPVDIR"/shadow-rmiss.spv"},
        .chitCount = 1,
        .chitShaders = (char*[]){SPVDIR"/shadow-rchit.spv"}
    };

    assert(OBDN_ARRAY_SIZE(gPipelineInfos) == GBUFFER_PIPELINE_COUNT);

    obdn_r_CreateGraphicsPipelines(OBDN_ARRAY_SIZE(gPipelineInfos), gPipelineInfos, gbufferPipelines);
    obdn_r_CreateGraphicsPipelines(1, &defferedPipeInfo, &defferedPipeline);
    obdn_r_CreateRayTracePipelines(1, &rtPipelineInfo, &raytracePipeline, &shaderBindingTable);
}

static void updateGbufferDescriptors(void)
{
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        VkDescriptorImageInfo worldPInfo = {
            .sampler     = imageWorldP.sampler,
            .imageView   = imageWorldP.view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL };

        VkDescriptorImageInfo normalInfo = {
            .sampler     = imageNormal.sampler,
            .imageView   = imageNormal.view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL };

        VkDescriptorImageInfo albedoInfo = {
            .sampler     = imageAlbedo.sampler,
            .imageView   = imageAlbedo.view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL };

        VkDescriptorImageInfo shadowInfo = {
            .sampler     = imageShadow.sampler,
            .imageView   = imageShadow.view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL };

        VkDescriptorImageInfo roughnessInfo = {
            .sampler     = imageRoughness.sampler,
            .imageView   = imageRoughness.view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL };

        VkWriteDescriptorSet writes[] = {{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstArrayElement = 0,
            .dstSet = descriptions[i].descriptorSets[DESC_SET_DEFERRED],
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo     = &worldPInfo
        },{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstArrayElement = 0,
            .dstSet = descriptions[i].descriptorSets[DESC_SET_DEFERRED],
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo     = &normalInfo 
        },{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstArrayElement = 0,
            .dstSet = descriptions[i].descriptorSets[DESC_SET_DEFERRED],
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo     = &albedoInfo
        },{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstArrayElement = 0,
            .dstSet = descriptions[i].descriptorSets[DESC_SET_DEFERRED],
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo     = &shadowInfo
        },{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstArrayElement = 0,
            .dstSet = descriptions[i].descriptorSets[DESC_SET_DEFERRED],
            .dstBinding = 4,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo     = &roughnessInfo
        }};

        vkUpdateDescriptorSets(device, OBDN_ARRAY_SIZE(writes), writes, 0, NULL);
    }
}

static void updateASDescriptors(void)
{
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        VkWriteDescriptorSetAccelerationStructureKHR asInfo = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            .accelerationStructureCount = 1,
            .pAccelerationStructures    = &tlas.handle
        };

        VkWriteDescriptorSet writeDS = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstArrayElement = 0,
            .dstSet = descriptions[i].descriptorSets[DESC_SET_DEFERRED],
            .dstBinding = 5,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            .pNext = &asInfo
        };

        vkUpdateDescriptorSets(device, 1, &writeDS, 0, NULL);
    }
}

static void updateDescriptors(void)
{
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
    {
        // camera creation
        cameraBuffers[i] = obdn_v_RequestBufferRegion(sizeof(Camera),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, OBDN_V_MEMORY_HOST_GRAPHICS_TYPE);

        // xforms creation
        xformsBuffers[i] = obdn_v_RequestBufferRegion(sizeof(Xforms), 
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, OBDN_V_MEMORY_HOST_GRAPHICS_TYPE);

        // lights creation 
        lightsBuffers[i] = obdn_v_RequestBufferRegion(sizeof(Lights), 
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, OBDN_V_MEMORY_HOST_GRAPHICS_TYPE);

        materialsBuffers[i] = obdn_v_RequestBufferRegion(sizeof(Material) * OBDN_S_MAX_MATERIALS, 
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, OBDN_V_MEMORY_HOST_GRAPHICS_TYPE);

        VkDescriptorBufferInfo camInfo = {
            .buffer = cameraBuffers[i].buffer,
            .offset = cameraBuffers[i].offset,
            .range  = cameraBuffers[i].size
        };

        VkDescriptorBufferInfo xformInfo = {
            .buffer = xformsBuffers[i].buffer,
            .offset = xformsBuffers[i].offset,
            .range  = xformsBuffers[i].size
        };

        VkDescriptorBufferInfo lightInfo = {
            .buffer = lightsBuffers[i].buffer,
            .offset = lightsBuffers[i].offset,
            .range  = lightsBuffers[i].size
        };

        VkDescriptorBufferInfo materialInfo = {
            .buffer = materialsBuffers[i].buffer,
            .offset = materialsBuffers[i].offset,
            .range  = materialsBuffers[i].size
        };

        VkWriteDescriptorSet writes[] = {{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstArrayElement = 0,
            .dstSet = descriptions[i].descriptorSets[DESC_SET_MAIN],
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &camInfo
        },{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstArrayElement = 0,
            .dstSet = descriptions[i].descriptorSets[DESC_SET_MAIN],
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &xformInfo
        },{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstArrayElement = 0,
            .dstSet = descriptions[i].descriptorSets[DESC_SET_MAIN],
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &lightInfo 
        },{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstArrayElement = 0,
            .dstSet = descriptions[i].descriptorSets[DESC_SET_MAIN],
            .dstBinding = 4,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &materialInfo
        }};

        vkUpdateDescriptorSets(device, OBDN_ARRAY_SIZE(writes), writes, 0, NULL);
    }

    updateGbufferDescriptors();
}

static void updateTexture(const uint32_t frameIndex, const Obdn_V_Image* img, const uint32_t texId)
{
    VkDescriptorImageInfo textureInfo = {
        .imageLayout = img->layout,
        .imageView   = img->view,
        .sampler     = img->sampler
    };

    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = NULL,
        .dstSet = descriptions[frameIndex].descriptorSets[DESC_SET_MAIN],
        .dstBinding = 3,
        .dstArrayElement = texId,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &textureInfo
    };

    vkUpdateDescriptorSets(device, 1, &write, 0, NULL);

    printf("Updated Texture %d frame %d\n", texId, frameIndex);
}

static void generateGBuffer(VkCommandBuffer cmdBuf, const uint32_t frameIndex)
{
    VkClearValue clearValueColor = {0.0f, 0.0f, 0.0f, 0.0f};
    VkClearValue clearValueMatid = {0};
    VkClearValue clearValueDepth = {1.0, 0};

    VkClearValue clears[] = {clearValueColor, clearValueColor, clearValueColor, clearValueMatid, clearValueDepth};

    VkRenderPassBeginInfo rpassInfo = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .clearValueCount = OBDN_ARRAY_SIZE(clears),
        .pClearValues = clears,
        .renderArea = {{0, 0}, {windowWidth, windowHeight}},
        .renderPass =  gbufferRenderPass,
        .framebuffer = gbuffers[frameIndex]
    };

    vkCmdBeginRenderPass(cmdBuf, &rpassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdPushConstants(cmdBuf, pipelineLayout, 
            VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR, sizeof(uint32_t) * 2, sizeof(uint32_t), &scene->lightCount);

    //assert(sizeof(Vec4) == sizeof(Obdn_S_Material));
    assert(scene->primCount < MAX_PRIM_COUNT);

    for (int pipeId = 0; pipeId < GBUFFER_PIPELINE_COUNT; pipeId++)
    {
        vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, gbufferPipelines[pipeId]);
        const uint32_t primCount = pipelinePrimLists[pipeId].primCount;
        printf("Pipeline %d, primCount %d\n", pipeId, primCount);
        for (int i = 0; i < primCount; i++)
        {
            Obdn_S_PrimId primId = pipelinePrimLists[pipeId].primIds[i];
            Obdn_S_MaterialId matId = scene->prims[primId].materialId;
            vkCmdPushConstants(cmdBuf, pipelineLayout, 
                    VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(uint32_t), &primId);
            vkCmdPushConstants(cmdBuf, pipelineLayout, 
                    VK_SHADER_STAGE_VERTEX_BIT, sizeof(uint32_t), sizeof(uint32_t), &matId);
            obdn_r_DrawPrim(cmdBuf, &scene->prims[primId].rprim);
        }
    }

    vkCmdEndRenderPass(cmdBuf);
}

static void shadowPass(VkCommandBuffer cmdBuf, const uint32_t frameIndex)
{
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, raytracePipeline);

    vkCmdTraceRaysKHR(
        cmdBuf, &shaderBindingTable.raygenTable, &shaderBindingTable.missTable,
        &shaderBindingTable.hitTable, &shaderBindingTable.callableTable,
        windowWidth, windowHeight, 1);
}

static void deferredRender(VkCommandBuffer cmdBuf, const uint32_t frameIndex)
{
    VkClearValue clearValueColor = {0.0f, 0.0f, 0.0f, 0.0f};

    VkRenderPassBeginInfo rpassInfo = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .clearValueCount = 1,
        .pClearValues = &clearValueColor,
        .renderArea = {{0, 0}, {windowWidth, windowHeight}},
        .renderPass =  deferredRenderPass,
        .framebuffer = swapImageBuffer[frameIndex]
    };

    vkCmdBeginRenderPass(cmdBuf, &rpassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, defferedPipeline);

    vkCmdDraw(cmdBuf, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmdBuf);
}

static void sortPipelinePrims(void)
{
    for (int i = 0; i < GBUFFER_PIPELINE_COUNT; i++)
    {
        obdn_s_ClearPrimList(&pipelinePrimLists[i]);
    }
    for (Obdn_S_PrimId primId = 0; primId < scene->primCount; primId++) 
    {
        const Obdn_R_Primitive* prim = &scene->prims[primId].rprim;
        if (strcmp(prim->attrNames[3], "tan") == 0)
            obdn_s_AddPrimToList(primId, &pipelinePrimLists[PIPELINE_GBUFFER_TAN]); 
        else
            obdn_s_AddPrimToList(primId, &pipelinePrimLists[PIPELINE_GBUFFER_NO_TAN]); 
        //if (mat->textureAlbedo && mat->textureRoughness && mat->textureNormal)
        //    addPrimToPipelinePrimList(primId, &pipelinePrimLists[PIPELINE_TAN]);
        //else if (mat->textureAlbedo && mat->textureRoughness)
        //    addPrimToPipelinePrimList(primId, &pipelinePrimLists[PIPELINE_REG]);
        //else if (mat->textureAlbedo + mat->textureNormal + mat->textureNormal == 0 )
        //    addPrimToPipelinePrimList(primId, &pipelinePrimLists[PIPELINE_NO_MAPS]);
        //else
        //    assert(0 && "currently prims must have albedo and roughness textures");
    }
}

static void updateRenderCommands(const uint32_t frameIndex)
{
    obdn_v_ResetCommand(&renderCommands[frameIndex]);

    VkCommandBuffer cmdBuf = renderCommands[frameIndex].buffer;

    obdn_v_BeginCommandBuffer(cmdBuf);

    vkCmdBindDescriptorSets(
        cmdBuf, 
        VK_PIPELINE_BIND_POINT_GRAPHICS, 
        pipelineLayout,
        0, 2, descriptions[frameIndex].descriptorSets,
        0, NULL);

    VkViewport viewport = {
        .width = windowWidth,
        .height = windowHeight,
        .minDepth = 0.0,
        .maxDepth = 1.0,
        .x = 0, .y = 0
    };

    VkRect2D scissor = {
        .extent = {windowWidth, windowHeight},
        .offset = {0, 0}
    };

    vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
    vkCmdSetScissor(cmdBuf, 0, 1, &scissor);

    generateGBuffer(cmdBuf, frameIndex);

    obdn_v_MemoryBarrier(
        cmdBuf, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    vkCmdBindDescriptorSets(
        cmdBuf, 
        VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, 
        pipelineLayout,
        0, 2, descriptions[frameIndex].descriptorSets,
        0, NULL);

    shadowPass(cmdBuf, frameIndex);

    obdn_v_MemoryBarrier(
        cmdBuf, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    vkCmdBindDescriptorSets(
        cmdBuf, 
        VK_PIPELINE_BIND_POINT_GRAPHICS, 
        pipelineLayout,
        0, 2, descriptions[frameIndex].descriptorSets,
        0, NULL);

    vkCmdSetViewport(cmdBuf, 0, 1, &viewport);

    deferredRender(cmdBuf, frameIndex);

    obdn_v_EndCommandBuffer(cmdBuf);
}

static void cleanUpSwapchainDependent(void)
{
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
    {
        vkDestroyFramebuffer(device, framebuffers[i], NULL);
        vkDestroyFramebuffer(device, gbuffers[i], NULL);
        vkDestroyFramebuffer(device, swapImageBuffer[i], NULL);
    }
    obdn_v_FreeImage(&renderTargetDepth);
    obdn_v_FreeImage(&imageWorldP);
    obdn_v_FreeImage(&imageNormal);
    obdn_v_FreeImage(&imageShadow);
    obdn_v_FreeImage(&imageRoughness);
    obdn_v_FreeImage(&imageAlbedo);
}

static void onSwapchainRecreate(void)
{
    vkDeviceWaitIdle(device);
    windowWidth  = scene->window[0];
    windowHeight = scene->window[1];
    cleanUpSwapchainDependent();
    initAttachments();
    initFramebuffers();
    updateGbufferDescriptors();
}

static void updateCamera(uint32_t index)
{
    const Mat4 proj = m_BuildPerspective(0.001, 100);
    const Mat4 view = m_Invert4x4(&scene->camera.xform);
    Camera* uboCam = (Camera*)cameraBuffers[index].hostData;
    uboCam->view = view;
    uboCam->proj = proj;
    uboCam->camera = scene->camera.xform;
}

static void updateXform(uint32_t frameIndex, uint32_t primIndex)
{
    Xforms* xforms = (Xforms*)xformsBuffers[frameIndex].hostData;
    xforms->xform[primIndex] = scene->xforms[primIndex];
}

static void updateLight(uint32_t frameIndex, uint32_t lightIndex)
{
    Lights* lights = (Lights*)lightsBuffers[frameIndex].hostData;
    lights->light[lightIndex] = scene->lights[lightIndex];
}

static void updateMaterials(uint32_t frameIndex)
{
    memcpy(materialsBuffers[frameIndex].hostData, scene->materials, sizeof(Material) * scene->materialCount);
}

static void buildAccelerationStructures(void)
{
    for (int i = 0; i < scene->primCount; i++)
    {
        AccelerationStructure* blas = &blasses[i];
        if (blas->bufferRegion.size != 0)
            obdn_r_DestroyAccelerationStruct(blas);
        obdn_r_BuildBlas(&scene->prims[i].rprim, blas);
    }
    if (tlas.bufferRegion.size != 0)
        obdn_r_DestroyAccelerationStruct(&tlas);
    obdn_r_BuildTlasNew(scene->primCount, blasses, scene->xforms, &tlas);
    printf(">>>>> Built acceleration structures\n");
}

static void syncScene(const uint32_t frameIndex)
{
    static uint8_t cameraNeedUpdate    = MAX_FRAMES_IN_FLIGHT;
    static uint8_t xformsNeedUpdate    = MAX_FRAMES_IN_FLIGHT;
    static uint8_t lightsNeedUpdate    = MAX_FRAMES_IN_FLIGHT;
    static uint8_t texturesNeedUpdate  = MAX_FRAMES_IN_FLIGHT;
    static uint8_t materialsNeedUpdate = MAX_FRAMES_IN_FLIGHT;
    static uint8_t framesNeedUpdate    = MAX_FRAMES_IN_FLIGHT;

    if (scene->dirt)
    {
        if (scene->dirt & OBDN_S_CAMERA_VIEW_BIT)
            cameraNeedUpdate = MAX_FRAMES_IN_FLIGHT;
        if (scene->dirt & OBDN_S_LIGHTS_BIT)
            lightsNeedUpdate = MAX_FRAMES_IN_FLIGHT;
        if (scene->dirt & OBDN_S_XFORMS_BIT)
            xformsNeedUpdate = MAX_FRAMES_IN_FLIGHT;
        if (scene->dirt & OBDN_S_MATERIALS_BIT)
            materialsNeedUpdate = MAX_FRAMES_IN_FLIGHT;
        if (scene->dirt & OBDN_S_TEXTURES_BIT)
        {
            texturesNeedUpdate = framesNeedUpdate = MAX_FRAMES_IN_FLIGHT;
        }
        if (scene->dirt & OBDN_S_PRIMS_BIT)
        {
            sortPipelinePrims();
            buildAccelerationStructures();
            updateASDescriptors();
            framesNeedUpdate = MAX_FRAMES_IN_FLIGHT;
        }
        if (scene->dirt & OBDN_S_WINDOW_BIT)
        {
            onSwapchainRecreate();
            framesNeedUpdate = MAX_FRAMES_IN_FLIGHT;
        }
    }
    if (cameraNeedUpdate)
    {
        updateCamera(frameIndex);
        cameraNeedUpdate--;
    }
    if (xformsNeedUpdate)
    {
        for (int i = 0; i < scene->primCount; i++) 
            updateXform(frameIndex, i);
        xformsNeedUpdate--;
    }
    if (lightsNeedUpdate)
    {
        for (int i = 0; i < scene->lightCount; i++) 
            updateLight(frameIndex, i);
        lightsNeedUpdate--;
    }
    if (materialsNeedUpdate)
    {
        updateMaterials(frameIndex);
        materialsNeedUpdate--;
    }
    if (texturesNeedUpdate) // TODO update all tex
    {
        printf("texturesNeedUpdate %d\n", texturesNeedUpdate);
        for (int i = 1; i <= scene->textureCount; i++)  // remember, 1 is the first valid texture index
        {
            updateTexture(frameIndex, &scene->textures[i].devImage, i);
        }
        texturesNeedUpdate--;
    }
    if (framesNeedUpdate)
    {
        updateRenderCommands(frameIndex);
        framesNeedUpdate--;
    }
}

void r_InitRenderer(const Obdn_S_Scene* scene_)
{
    scene = scene_;

    windowWidth = scene_->window[0];
    windowHeight = scene_->window[1];

    memset(pipelinePrimLists, 0, sizeof(pipelinePrimLists));

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
    {
        renderCommands[i] = obdn_v_CreateCommand(OBDN_V_QUEUE_GRAPHICS_TYPE);
    }

    initAttachments();
    V1_PRINT(">> Tanto: attachments initialized. \n");
    initRenderPass();
    V1_PRINT(">> Tanto: renderpasses initialized. \n");
    initFramebuffers();
    V1_PRINT(">> Tanto: framebuffers initialized. \n");
    initDescriptorSetsAndPipelineLayouts();
    V1_PRINT(">> Tanto: descriptor sets and pipeline layouts initialized. \n");
    updateDescriptors();
    V1_PRINT(">> Tanto: descriptors updated. \n");
    initPipelines();
    V1_PRINT(">> Tanto: pipelines initialized. \n");
    V1_PRINT(">> Tanto: initialization complete. \n");
}

VkSemaphore r_Render(uint32_t f, VkSemaphore waitSemephore)
{
    assert(scene->primCount);
    obdn_v_WaitForFence(&renderCommands[f].fence);
    syncScene(f);
    obdn_v_SubmitGraphicsCommand(0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, waitSemephore, renderCommands[f].semaphore, renderCommands[f].fence, renderCommands[f].buffer);
    waitSemephore = renderCommands[f].semaphore;
    if (ri.renderUi)
        waitSemephore = ri.renderUi(waitSemephore);
    if (ri.presentFrame)
        ri.presentFrame(waitSemephore);
    return waitSemephore;
}

void r_CleanUp(void)
{
    cleanUpSwapchainDependent();
    for (int i = 0; i < GBUFFER_PIPELINE_COUNT; i++) 
    {
        vkDestroyPipeline(device, gbufferPipelines[i], NULL);
    }
    vkDestroyPipeline(device, defferedPipeline, NULL);
    vkDestroyPipeline(device, raytracePipeline, NULL);
    for (int i = 0; i < OBDN_S_MAX_PRIMS; i++)
    {
        AccelerationStructure* blas = &blasses[i];
        if (blas->bufferRegion.size != 0)
            obdn_r_DestroyAccelerationStruct(blas);
    }
    if (tlas.bufferRegion.size != 0)
        obdn_r_DestroyAccelerationStruct(&tlas);
    obdn_r_DestroyShaderBindingTable(&shaderBindingTable);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        obdn_v_DestroyCommand(renderCommands[i]);
        obdn_r_DestroyDescription(&descriptions[i]);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayouts[i], NULL);
        obdn_v_FreeBufferRegion(&cameraBuffers[i]);
        obdn_v_FreeBufferRegion(&xformsBuffers[i]);
        obdn_v_FreeBufferRegion(&lightsBuffers[i]);
        obdn_v_FreeBufferRegion(&materialsBuffers[i]);
    }
    vkDestroyRenderPass(device, renderpass, NULL);
    vkDestroyRenderPass(device, gbufferRenderPass, NULL);
    vkDestroyRenderPass(device, deferredRenderPass, NULL);
    vkDestroyPipelineLayout(device, pipelineLayout, NULL);
}
