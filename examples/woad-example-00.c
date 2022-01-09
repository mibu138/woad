#include <woad/woad.h>
#include <hell/hell.h>
#include <obsidian/obsidian.h>

Hell_Hellmouth hm;
Obdn_Orb orb;
Obdn_Swapchain* swapchain;

void frame(u64 fi, u64 dt)
{
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

    hell_Loop(&hm);
    hell_CloseHellmouth(&hm);
    return 0;
}
