#ifndef WOAD_WOAD_H
#define WOAD_WOAD_H

#ifdef __cplusplus
extern "C" {
#endif

#include <obsidian/obsidian.h>

typedef enum {
    WOAD_SETTINGS_NO_RAYTRACE_BIT = 1 << 1
} Woad_Settings_Flags;

void
woad_Init(const Obdn_Instance* instance, Obdn_Memory* memory,
                  VkImageLayout finalColorLayout,
                  VkImageLayout finalDepthLayout, uint32_t fbCount,
                  const Obdn_Frame fbs[/*fbCount*/],
                  Woad_Settings_Flags flags);
void
woad_Render(const Obdn_Scene* scene, const Obdn_Frame* fb, uint32_t x, uint32_t y, uint32_t width,
                  uint32_t height, VkCommandBuffer cmdbuf);

void
woad_Cleanup(void);

#ifdef __cplusplus
}
#endif

#endif // WOAD_WOAD_H
