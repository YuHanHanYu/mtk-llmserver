#pragma once

#include <string>

namespace mtk::backend {

// Specify custom Neuron Adapter library path.
// Returns true if the library path can be set (i.e. Neuron Library has not yet been loaded),
// or returns false if otherwise.
bool set_neuron_adapter_lib_path(const std::string& libPath);

namespace neuron_api {

bool load_library();

} // namespace neuron_api

} // namespace mtk::backend