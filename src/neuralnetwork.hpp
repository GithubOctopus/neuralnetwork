#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "neuron.hpp"
#include "neuronlayer.hpp"

namespace NN {
class NeuralNetwork {
private:
  std::vector<NeuronLayer*> layers;
  const std::function<float(float)> &activation_function;
  const std::function<float(float)> &activation_function_derivative;
  float learning_rate;
public:
  std::vector<float> train(
    std::vector<float> input,
    std::vector<float> ideal_output
  );
  std::vector<float> run(std::vector<float> input);
  int result();
  NeuralNetwork(
    int num_layers,
    int* layer_sizes,
    float learning_rate,
    const std::function<float(float)> &activation_function = sigmoidFunction,
    const std::function<float(float)> &activation_function_derivative = sigmoidDerivativeFunction,
    const std::function<float()> &generate_bias = [](){return 0.0f;},
    const std::function<std::vector<float>(int)> &generate_weights = glorotInitialize
  );
};


}
#endif
