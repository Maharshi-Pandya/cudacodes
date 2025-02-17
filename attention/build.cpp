#include <cublas.h>
#include <torch/extension.h>

torch::Tensor attention_forward(uint64_t handle, torch::Tensor Q, torch::Tensor K, torch::Tensor V);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward", torch::wrap_pybind_function(attention_forward), "attention_forward");
}
