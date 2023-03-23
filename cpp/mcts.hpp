#pragma once

#include <torch/script.h> // One-stop header.
#include <vector>

std::vector<torch::Tensor> infer(std::string model_path, torch::Tensor features);