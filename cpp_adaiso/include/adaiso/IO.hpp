#pragma once

#include "adaiso/Mesh.hpp"

#include <filesystem>

namespace adaiso {

[[nodiscard]] Mesh loadMesh(const std::filesystem::path& path);
[[nodiscard]] Mesh loadObj(const std::filesystem::path& path);
[[nodiscard]] Mesh loadAsciiPly(const std::filesystem::path& path);
[[nodiscard]] Mesh loadOff(const std::filesystem::path& path);
void writeObj(const std::filesystem::path& path, const Mesh& mesh);

}  // namespace adaiso
