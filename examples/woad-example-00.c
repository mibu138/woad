#define COAL_SIMPLE_TYPE_NAMES
#include <coal/coal.h>
#include <hell/hell.h>
#include <obsidian/obsidian.h>
#include <woad/woad.h>

Hell_Hellmouth  hm;
Obdn_Orb        orb;
Obdn_Swapchain* swapchain;

Obdn_Scene* scene;
Obdn_Geometry geo;

Obdn_CommandPool cmdpool;

VkFence     fences[2];
VkSemaphore img_acq_semas[2];
VkSemaphore rendered_semas[2];

bool
handlePointerInput(const Hell_Event* event, void* data)
{
    static bool lmbdown = false, mmbdown = false, rmbdown = false;
    static int  mx = 0, my = 0;
    static Vec3 target = {0, 0, 0};
    switch (event->type)
    {
    case HELL_EVENT_TYPE_MOUSEUP:
        lmbdown = false;
        break;
    case HELL_EVENT_TYPE_MOUSEDOWN:
        lmbdown = true;
        break;
    default:
        break;
    }
    obdn_UpdateCamera_ArcBall(
        scene, &target, hell_GetWindowWidth(hm.windows[0]),
        hell_GetWindowHeight(hm.windows[0]), 0.16, mx, hell_GetMouseX(event),
        my, hell_GetMouseY(event), false, lmbdown, false, false);
    mx = hell_GetMouseX(event);
    my = hell_GetMouseY(event);
    return true;
}

void
frame(i64 fi, i64 dt)
{
    u32 f = fi % 2;

    VkCommandBuffer   cmdbuf = cmdpool.cmdbufs[f];
    const Obdn_Frame* frame =
        obdn_AcquireSwapchainFrame(swapchain, 0, img_acq_semas[f]);

    obdn_WaitForFence(orb.device, &fences[f]);
    vkResetCommandBuffer(cmdbuf, 0);

    // Mat4 cam = obdn_SceneGetCameraXform(scene);
    // coal_PrintMat4(cam);

    obdn_BeginCommandBuffer(cmdbuf);

    woad_Render(scene, frame, 0, 0, frame->width, frame->height, cmdbuf);

    obdn_EndCommandBuffer(cmdbuf);

    VkSubmitInfo si = obdn_SubmitInfo(
        1, &img_acq_semas[f],
        &(VkPipelineStageFlags){VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT}, 1, &cmdbuf,
        1, &rendered_semas[f]);

    VkQueue queue = obdn_GetGraphicsQueue(orb.instance, 0);

    vkQueueSubmit(queue, 1, &si, fences[f]);

    obdn_PresentFrame(swapchain, 1, &rendered_semas[f]);

    obdn_SceneEndFrame(scene);
}

int
main(int argc, char* argv[])
{
    hell_Print("Starting\n");
    hell_OpenHellmouth(frame, NULL, &hm);
    Obdn_InstanceParms ip = {.enableRayTracing = false,
                             .surfaceType      = OBDN_SURFACE_TYPE_XCB};
    obdn_CreateOrb(&ip, 50, 50, 200, 0, 0, &orb);
    swapchain = obdn_AllocSwapchain();
    hell_HellmouthAddWindow(&hm, 500, 500, NULL);
    Obdn_AovInfo depth_aov = {.aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT,
                              .format      = VK_FORMAT_D32_SFLOAT,
                              .usageFlags =
                                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT};
    obdn_CreateSwapchain(orb.instance, orb.memory, hm.eventqueue, hm.windows[0],
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 1, &depth_aov,
                         swapchain);
    woad_Init(orb.instance, orb.memory, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 2,
              obdn_GetSwapchainFrames(swapchain), WOAD_SETTINGS_NO_RAYTRACE_BIT);

    scene = obdn_AllocScene();
    obdn_CreateScene(hm.grimoire, orb.memory, 1, 1, 0.01, 100, scene);

    geo = obdn_CreateCube(orb.memory, false);

    obdn_SceneAddPrim(scene, &geo, COAL_MAT4_IDENT, (Obdn_MaterialHandle){0});

    obdn_SceneAddPointLight(scene, (Coal_Vec3){0, 2, 0}, (Coal_Vec3){1, 1, 1},
                            1.0);
    obdn_SceneAddPointLight(scene, (Coal_Vec3){1, 2, 2.5}, (Coal_Vec3){1, 1, 1},
                            1.0);
    obdn_SceneAddPointLight(scene, (Coal_Vec3){2, 2, 1}, (Coal_Vec3){1, 1, 1},
                            1.0);
    obdn_SceneAddPointLight(scene, (Coal_Vec3){-1, -1, -2},
                            (Coal_Vec3){1, 1, 1}, 1.0);

    cmdpool = obdn_CreateCommandPool(
        orb.device,
        obdn_GetQueueFamilyIndex(orb.instance, OBDN_V_QUEUE_GRAPHICS_TYPE),
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 2);

    obdn_CreateFences(orb.device, true, 2, fences);
    obdn_CreateSemaphores(orb.device, 2, img_acq_semas);
    obdn_CreateSemaphores(orb.device, 2, rendered_semas);

    hell_Subscribe(hm.eventqueue, HELL_EVENT_MASK_POINTER_BIT,
                   hell_GetWindowID(hm.windows[0]), handlePointerInput, NULL);
    hell_Loop(&hm);
    hell_CloseHellmouth(&hm);
    return 0;
}
