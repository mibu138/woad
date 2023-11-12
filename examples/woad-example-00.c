#define COAL_SIMPLE_TYPE_NAMES
#include <coal/coal.h>
#include <hell/hell.h>
#include <woad/woad.h>

#include <unistd.h>

Hell hm;
Onyx orb;
OnyxSwapchain *swapchain;

OnyxScene* scene;
OnyxGeometry geo;

OnyxCommandPool cmdpool;

VkFence     fences[2];
VkSemaphore rendered_semas[2];

HellWindow *window;

bool
handlePointerInput(const HellEvent* event, void* data)
{
    static bool lmbdown = false, mmbdown = false, rmbdown = false;
    static int  mx = 0, my = 0;
    static Vec3 target = {0, 0, 0};
    uint8_t button;
    switch (event->type)
    {
    case HELL_EVENT_TYPE_MOUSEUP:
        button = hell_get_event_button_code(event);
        printf("button %d\n", button);
        if (button == HELL_MOUSE_LEFT)
            lmbdown = false;
        if (button == HELL_MOUSE_RIGHT)
            rmbdown = false;
        if (button == HELL_MOUSE_MID)
            mmbdown = false;
        break;
    case HELL_EVENT_TYPE_MOUSEDOWN:
        button = hell_get_event_button_code(event);
        printf("button %d\n", button);
        if (button == HELL_MOUSE_LEFT)
            lmbdown = true;
        if (button == HELL_MOUSE_RIGHT)
            rmbdown = true;
        if (button == HELL_MOUSE_MID)
            mmbdown = true;
        break;
    default:
        break;
    }
    onyx_update_camera_arc_ball(
        scene, &target, hell_get_window_width(hm.windows[0]),
        hell_get_window_height(hm.windows[0]), 0.016, mx, hell_get_mouse_x(event),
        my, hell_get_mouse_y(event), mmbdown, lmbdown, rmbdown, false);
    mx = hell_get_mouse_x(event);
    my = hell_get_mouse_y(event);
    return true;
}

void
frame(i64 fi, i64 dt)
{
    u32 f = fi % 2;

    VkCommandBuffer   cmdbuf = cmdpool.cmdbufs[f];
    OnyxSwapchainImage swap_img =
        onyx_acquire_swapchain_image(swapchain);

    onyx_wait_for_fence(orb.device, &fences[f]);
    vkResetCommandBuffer(cmdbuf, 0);

    // Mat4 cam = obdn_SceneGetCameraXform(scene);
    // coal_PrintMat4(cam);
    unsigned int scwidth = onyx_get_swapchain_width(swapchain);
    unsigned int scheight = onyx_get_swapchain_height(swapchain);

    WoadFrame frame = woad_Frame(&swap_img);

    onyx_begin_command_buffer(cmdbuf);

    woad_Render(scene, &frame, 0, 0, scwidth, scheight, cmdbuf);

    onyx_end_command_buffer(cmdbuf);

    VkSubmitInfo si = onyx_submit_info(
        1, &swap_img.swapchain->image_acquired[swap_img.semaphore_index],
        &(VkPipelineStageFlags){VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT}, 1, &cmdbuf,
        1, &rendered_semas[f]);

    VkQueue queue = onyx_get_graphics_queue(orb.instance, 0);

    vkQueueSubmit(queue, 1, &si, fences[f]);

    onyx_present_swapchains(queue,
            1, &swapchain, 1, &rendered_semas[f]);

    onyx_scene_end_frame(scene);
}

int
main(int argc, char* argv[])
{
    hell_print("Starting\n");
    hell_open_mouth(0, &hm);
    OnyxInstanceParms ip = {.enable_ray_tracing = false,
                             .surface_type = ONYX_SURFACE_TYPE_XCB};
    onyx_create_orb(&ip, 50, 50, 200, 0, 0, &orb);
    swapchain = onyx_alloc_swapchain();
    window = hell_hellmouth_add_window(&hm, 500, 500, NULL);
    onyx_create_swapchain(orb.instance->vkinstance, orb.instance->device, orb.instance->physical_device, window, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                         swapchain);
    woad_Init(orb.instance, orb.memory, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
              swapchain, WOAD_SETTINGS_NO_RAYTRACE_BIT);

    scene = onyx_alloc_scene();
    onyx_create_scene(hm.grimoire, orb.memory, 1, 1, 0.01, 100, scene);

    geo = onyx_create_cube(orb.memory, false);

    onyx_scene_add_prim(scene, &geo, COAL_MAT4_IDENT, (OnyxMaterialHandle){0});

    onyx_scene_add_point_light(scene, (CoalVec3){0, 2, 0}, (CoalVec3){1, 1, 1},
    1.0);
    onyx_scene_add_point_light(scene, (CoalVec3){1, 2, 2.5}, (CoalVec3){1, 1, 1},
    1.0);
    onyx_scene_add_point_light(scene, (CoalVec3){2, 2, 1}, (CoalVec3){1, 1, 1},
    1.0);
    onyx_scene_add_point_light(scene, (CoalVec3){-1, -1, -2},
                            (CoalVec3){1, 1, 1}, 1.0);

    cmdpool = onyx_create_command_pool_(
        orb.device,
        onyx_queue_family_index(orb.instance, ONYX_QUEUE_GRAPHICS_TYPE),
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 2);

    onyx_create_fences(orb.device, true, 2, fences);
    onyx_create_semaphores(orb.device, 2, rendered_semas);

    hell_subscribe(hm.eventqueue, HELL_EVENT_MASK_POINTER_BIT,
                   hell_get_window_i_d(hm.windows[0]), handlePointerInput, NULL);
    hell_loop(&hm, frame);
    hell_close_hellmouth(&hm);
    return 0;
}
