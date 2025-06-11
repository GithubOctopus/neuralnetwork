#ifndef NEURONLAYER_HPP
#define NEURONLAYER_HPP

#include "neuron.hpp"

namespace NN {

std::vector<float> glorotInitialize(int size);

class NeuronLayer {
private:
  std::vector<Neuron> neurons;
  NeuronLayer *parent;
public:
  NeuronLayer(
    int size,
    NeuronLayer *parent,
    std::function<float()> generate_bias,
    std::function<std::vector<float>(int)> generate_weights
  );
  void activate(std::function<float(float)> activation_function);
  std::vector<Neuron> *getNeuronsPtr();
  NeuronLayer *getParentPointer();
  int size();
  int parentSize();
};

}
#endif
