// SPDX-License-Identifier: MIT
#include <cstddef>
#include <mlx/memory.h>

// Observe before any residency mutation. Never emit MLX exception payloads.
extern "C" int stark_glm_mlx_get_wired_limit(std::size_t *result) noexcept {
    if (!result) return 1;
    try {
        *result = mlx::core::get_wired_limit();
        return 0;
    } catch (...) {
        return 1;
    }
}
