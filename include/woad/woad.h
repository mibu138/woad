#ifndef WOAD_WOAD_H
#define WOAD_WOAD_H

#ifdef __cplusplus
extern "C" {
#endif

#include <onyx/onyx.h>

typedef enum {
    WOAD_SETTINGS_NO_RAYTRACE_BIT = 1 << 1
} Woad_Settings_Flags;

typedef struct WoadFrame {
    VkImageView view;
    VkFormat    format;
    uint32_t    width;
    uint32_t    height;
    bool        dirty;
    uint8_t     index;
} WoadFrame;

WoadFrame
woad_Frame(const OnyxSwapchainImage *img);

void
woad_Init(const OnyxInstance* instance, OnyxMemory* memory,
                  VkImageLayout finalColorLayout,
                  VkImageLayout finalDepthLayout,
                  const OnyxSwapchain *swapchain,
                  Woad_Settings_Flags flags);
void
woad_Render(const OnyxScene* scene, const WoadFrame *fb, uint32_t x, uint32_t y, uint32_t width,
                  uint32_t height, VkCommandBuffer cmdbuf);

void
woad_Cleanup(void);

#ifdef __cplusplus
}
#endif

#endif // WOAD_WOAD_H
