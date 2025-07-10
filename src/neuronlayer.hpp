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
    const std::function<float()> &generate_bias = [](){return 0.0f;},
    const std::function<std::vector<float>(int)> &generate_weights = glorotInitialize
  );
  NeuronLayer(int size);
  NeuronLayer();
  void activate(const std::function<float(float)> &activation_function = sigmoidFunction);
  std::vector<Neuron> *getNeuronsPtr();
  NeuronLayer *getParentPtr();
  int size();
  int parentSize();
  std::pair<Neuron*, int> mostActivated();
};

}
#endif
