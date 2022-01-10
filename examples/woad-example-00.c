#include <woad/woad.h>
#include <hell/hell.h>
#include <obsidian/obsidian.h>

Hell_Hellmouth hm;
Obdn_Orb orb;
Obdn_Swapchain* swapchain;

Obdn_Scene* scene;

Obdn_CommandPool cmdpool;

VkFence     fences[2];
VkSemaphore img_acq_semas[2];
VkSemaphore rendered_semas[2];

void frame(u64 fi, u64 dt)
{
    u32 f = fi % 2;

    VkCommandBuffer cmdbuf = cmdpool.cmdbufs[f];
    const Obdn_Frame* frame = obdn_AcquireSwapchainFrame(swapchain, 0, img_acq_semas[f]);

    obdn_WaitForFence(orb.device, &fences[f]);
    vkResetCommandBuffer(cmdbuf, 0);

    obdn_BeginCommandBuffer(cmdbuf);

    woad_Render(scene, frame, cmdbuf);

    obdn_EndCommandBuffer(cmdbuf);

    VkSubmitInfo si = obdn_SubmitInfo(
            1, &img_acq_semas[f], 
            &(VkPipelineStageFlags){VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT},
            1, &cmdbuf,
            1, &rendered_semas[f]);

    VkQueue queue = obdn_GetGrahicsQueue(orb.instance, 0);

    vkQueueSubmit(queue, 1, &si, fences[f]);

    obdn_PresentFrame(swapchain, 1, &rendered_semas[f]);

    obdn_SceneEndFrame(scene);
}

int main(int argc, char *argv[])
{
    hell_Print("Starting\n");
    hell_OpenHellmouth(frame, NULL, &hm);
    Obdn_InstanceParms ip = {
        .enableRayTracing = true,
        .surfaceType = OBDN_SURFACE_TYPE_XCB
    };
    obdn_CreateOrb(&ip, 50, 50, 100, 0, 0, &orb);
    swapchain = obdn_AllocSwapchain();
    hell_HellmouthAddWindow(&hm, 500, 500, NULL);
    Obdn_AovInfo depth_aov = {
        .aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT,
        .format = VK_FORMAT_D32_SFLOAT,
        .usageFlags = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
    };
    obdn_CreateSwapchain(orb.instance, orb.memory, hm.eventqueue, hm.windows[0], 
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 1, &depth_aov, swapchain);
    woad_Init(orb.instance, orb.memory, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, 
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 2, obdn_GetSwapchainFrames(swapchain));

    scene = obdn_AllocScene();
    obdn_CreateScene(hm.grimoire, orb.memory, 1, 1, 0.01, 100, scene);

    cmdpool = obdn_CreateCommandPool(orb.device, obdn_GetQueueFamilyIndex(orb.instance, OBDN_V_QUEUE_GRAPHICS_TYPE),
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 2);

    obdn_CreateFences(orb.device, true, 2, fences);
    obdn_CreateSemaphores(orb.device, 2, img_acq_semas);
    obdn_CreateSemaphores(orb.device, 2, rendered_semas);

    hell_Loop(&hm);
    hell_CloseHellmouth(&hm);
    return 0;
}
