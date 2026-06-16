#include "adaiso/IO.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace adaiso {
namespace {

std::string lowerString(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::string trim(std::string value) {
  auto is_space = [](unsigned char c) {
    return std::isspace(c) != 0;
  };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](char c) {
                return !is_space(static_cast<unsigned char>(c));
              }));
  value.erase(std::find_if(value.rbegin(), value.rend(), [&](char c) {
                return !is_space(static_cast<unsigned char>(c));
              }).base(),
              value.end());
  return value;
}

bool nextDataLine(std::istream& in, std::string& line) {
  while (std::getline(in, line)) {
    const std::size_t hash = line.find('#');
    if (hash != std::string::npos) {
      line.erase(hash);
    }
    line = trim(line);
    if (!line.empty()) {
      return true;
    }
  }
  return false;
}

int parseObjIndex(const std::string& token, std::size_t vertex_count) {
  const std::size_t slash = token.find('/');
  const std::string head = token.substr(0, slash);
  if (head.empty()) {
    throw std::runtime_error("OBJ face token is missing a vertex index");
  }
  const int raw = std::stoi(head);
  int idx = 0;
  if (raw > 0) {
    idx = raw - 1;
  } else if (raw < 0) {
    idx = static_cast<int>(vertex_count) + raw;
  } else {
    throw std::runtime_error("OBJ indices are 1-based; index 0 is invalid");
  }
  if (idx < 0 || static_cast<std::size_t>(idx) >= vertex_count) {
    throw std::runtime_error("OBJ face contains an out-of-range vertex index");
  }
  return idx;
}

void appendFanTriangles(const std::vector<int>& polygon, std::vector<Eigen::Vector3i>& faces) {
  if (polygon.size() < 3) {
    return;
  }
  for (std::size_t i = 1; i + 1 < polygon.size(); ++i) {
    faces.emplace_back(polygon[0], polygon[i], polygon[i + 1]);
  }
}

}  // namespace

Mesh loadMesh(const std::filesystem::path& path) {
  const std::string ext = lowerString(path.extension().string());
  if (ext == ".obj") {
    return loadObj(path);
  }
  if (ext == ".ply") {
    return loadAsciiPly(path);
  }
  if (ext == ".off") {
    return loadOff(path);
  }
  throw std::runtime_error("unsupported mesh format: " + path.string());
}

Mesh loadObj(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open OBJ file: " + path.string());
  }

  Mesh mesh;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::string tag;
    ss >> tag;
    if (tag == "v") {
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      if (!(ss >> x >> y >> z)) {
        throw std::runtime_error("malformed OBJ vertex line in: " + path.string());
      }
      mesh.vertices.emplace_back(x, y, z);
    } else if (tag == "f") {
      std::vector<int> polygon;
      std::string token;
      while (ss >> token) {
        polygon.push_back(parseObjIndex(token, mesh.vertices.size()));
      }
      appendFanTriangles(polygon, mesh.faces);
    }
  }

  mesh.validate();
  return mesh;
}

Mesh loadAsciiPly(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open PLY file: " + path.string());
  }

  std::string line;
  if (!std::getline(in, line) || line != "ply") {
    throw std::runtime_error("not a PLY file: " + path.string());
  }

  bool ascii = false;
  std::size_t vertex_count = 0;
  std::size_t face_count = 0;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::string word;
    ss >> word;
    if (word == "format") {
      std::string format;
      ss >> format;
      ascii = (format == "ascii");
    } else if (word == "element") {
      std::string element;
      std::size_t count = 0;
      ss >> element >> count;
      if (element == "vertex") {
        vertex_count = count;
      } else if (element == "face") {
        face_count = count;
      }
    } else if (word == "end_header") {
      break;
    }
  }

  if (!ascii) {
    throw std::runtime_error("only ASCII PLY input is supported: " + path.string());
  }

  Mesh mesh;
  mesh.vertices.reserve(vertex_count);
  mesh.faces.reserve(face_count);

  for (std::size_t i = 0; i < vertex_count; ++i) {
    if (!std::getline(in, line)) {
      throw std::runtime_error("unexpected end of PLY vertex data: " + path.string());
    }
    std::istringstream ss(line);
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!(ss >> x >> y >> z)) {
      throw std::runtime_error("malformed PLY vertex line in: " + path.string());
    }
    mesh.vertices.emplace_back(x, y, z);
  }

  for (std::size_t i = 0; i < face_count; ++i) {
    if (!std::getline(in, line)) {
      throw std::runtime_error("unexpected end of PLY face data: " + path.string());
    }
    std::istringstream ss(line);
    int n = 0;
    ss >> n;
    if (n < 0) {
      throw std::runtime_error("malformed PLY face list in: " + path.string());
    }
    std::vector<int> polygon;
    polygon.reserve(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
      int idx = 0;
      if (!(ss >> idx)) {
        throw std::runtime_error("malformed PLY face index list in: " + path.string());
      }
      if (idx < 0 || static_cast<std::size_t>(idx) >= mesh.vertices.size()) {
        throw std::runtime_error("PLY face contains an out-of-range vertex index");
      }
      polygon.push_back(idx);
    }
    appendFanTriangles(polygon, mesh.faces);
  }

  mesh.validate();
  return mesh;
}

Mesh loadOff(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open OFF file: " + path.string());
  }

  std::string line;
  if (!nextDataLine(in, line)) {
    throw std::runtime_error("empty OFF file: " + path.string());
  }

  std::istringstream header(line);
  std::string magic;
  header >> magic;
  if (magic != "OFF") {
    throw std::runtime_error("not an OFF file: " + path.string());
  }

  std::size_t vertex_count = 0;
  std::size_t face_count = 0;
  std::size_t edge_count = 0;
  if (!(header >> vertex_count >> face_count >> edge_count)) {
    if (!nextDataLine(in, line)) {
      throw std::runtime_error("missing OFF counts line: " + path.string());
    }
    std::istringstream counts(line);
    if (!(counts >> vertex_count >> face_count >> edge_count)) {
      throw std::runtime_error("malformed OFF counts line: " + path.string());
    }
  }

  Mesh mesh;
  mesh.vertices.reserve(vertex_count);
  mesh.faces.reserve(face_count);

  for (std::size_t i = 0; i < vertex_count; ++i) {
    if (!nextDataLine(in, line)) {
      throw std::runtime_error("unexpected end of OFF vertex data: " + path.string());
    }
    std::istringstream ss(line);
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!(ss >> x >> y >> z)) {
      throw std::runtime_error("malformed OFF vertex line in: " + path.string());
    }
    mesh.vertices.emplace_back(x, y, z);
  }

  for (std::size_t i = 0; i < face_count; ++i) {
    if (!nextDataLine(in, line)) {
      throw std::runtime_error("unexpected end of OFF face data: " + path.string());
    }
    std::istringstream ss(line);
    int n = 0;
    if (!(ss >> n) || n < 0) {
      throw std::runtime_error("malformed OFF face line in: " + path.string());
    }
    std::vector<int> polygon;
    polygon.reserve(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
      int idx = 0;
      if (!(ss >> idx)) {
        throw std::runtime_error("malformed OFF face index list in: " + path.string());
      }
      if (idx < 0 || static_cast<std::size_t>(idx) >= mesh.vertices.size()) {
        throw std::runtime_error("OFF face contains an out-of-range vertex index");
      }
      polygon.push_back(idx);
    }
    appendFanTriangles(polygon, mesh.faces);
  }

  mesh.validate();
  return mesh;
}

void writeObj(const std::filesystem::path& path, const Mesh& mesh) {
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write OBJ file: " + path.string());
  }
  out << std::setprecision(17);
  for (const auto& v : mesh.vertices) {
    out << "v " << v[0] << ' ' << v[1] << ' ' << v[2] << '\n';
  }
  for (const auto& f : mesh.faces) {
    out << "f " << (f[0] + 1) << ' ' << (f[1] + 1) << ' ' << (f[2] + 1) << '\n';
  }
}

}  // namespace adaiso
