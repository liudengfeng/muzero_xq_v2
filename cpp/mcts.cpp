#include "mcts.hpp"

std::vector<torch::Tensor> infer(std::string model_path, torch::Tensor features)
{
    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(model_path);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
    }
    return module.initial_inference(features);
}