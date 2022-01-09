#ifndef WOAD_WOAD_H
#define WOAD_WOAD_H

#include <obsidian/obsidian.h>

void
woad_Init(Obdn_Instance* instance, Obdn_Memory* memory,
                  VkImageLayout finalColorLayout,
                  VkImageLayout finalDepthLayout, uint32_t fbCount,
                  const Obdn_Frame fbs[/*fbCount*/]);
void
woad_Render(const Obdn_Scene* scene, const Obdn_Frame* fb, VkCommandBuffer cmdbuf);

void
woad_Cleanup(void);

#endif // WOAD_WOAD_H
