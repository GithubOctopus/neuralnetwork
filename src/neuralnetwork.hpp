#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "neuron.hpp"
#include "neuronlayer.hpp"
#include <ostream>
#include <functional>

namespace NN {
class NeuralNetwork {
private:
  std::vector<NeuronLayer*> layers;
  const std::function<float(float)> &activation_function;
  const std::function<float(float)> &activation_function_derivative;
  const std::function<float()> &generate_bias;
  const std::function<std::vector<float>(int)> &generate_weights;
  float learning_rate;
public:
  std::vector<float> train(
    std::vector<float> input,
    std::vector<float> ideal_output
  );
  std::vector<float> run(std::vector<float> input);
  int result();
  void resetWeightsAndBiases();
  float getLearningRate();
  void setLearningRate(float new_learning_rate);
  NeuralNetwork(
    int num_layers,
    int* layer_sizes,
    float learning_rate,
    const std::function<float(float)> &activation_function = sigmoidFunction,
    const std::function<float(float)> &activation_function_derivative = sigmoidDerivativeFunction,
    const std::function<float()> &generate_bias = [](){return 0.0f;},
    const std::function<std::vector<float>(int)> &generate_weights = glorotInitialize
  );
  bool writeToFile(std::ostream &o) const;
};


}
#endif
