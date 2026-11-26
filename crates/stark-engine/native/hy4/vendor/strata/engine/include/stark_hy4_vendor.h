#pragma once

namespace stark::hy4 {

// Upstream's process-wide tuning and fault-injection options belong to its
// standalone tools. The embedded graph uses their defaults and typed inputs.
constexpr const char* vendor_option(const char*) noexcept { return nullptr; }

}  // namespace stark::hy4
