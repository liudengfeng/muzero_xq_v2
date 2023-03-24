#pragma once

#include <torch/script.h> // One-stop header.
#include <vector>

// std::vector<torch::Tensor> infer(std::string model_path, torch::Tensor features);
torch::jit::script::Module infer(std::string model_path);