#include "neuronlayer.hpp"
#include "neuron.hpp"
#include <cmath>

using namespace NN;

std::vector<float> NN::glorotInitialize(int size) {
  float limit = std::sqrt(6.0f / (size + 1));
  std::vector<float> result(size);
  for (int i = 0; i < size; i++) {
    float r = static_cast<float>(rand()) / RAND_MAX;
    result[i] = (r * 2 - 1) * limit;
  }
  return result;
}


NeuronLayer::NeuronLayer(int size) {
  this->neurons = std::vector<Neuron>(size);
  this->parent = nullptr;
}

NeuronLayer::NeuronLayer(
  int size,
  NeuronLayer *parent,
  std::function<float()> generate_bias,
  std::function<std::vector<float>(int)> generate_weights
) : 
  parent(parent),
  neurons(std::vector<Neuron>(size))
{
    for (int i = 0; i < size; i ++) {
      this->neurons[i] = Neuron(
        parent->getNeuronsPtr(), 
        generate_weights(parent->size()),
        generate_bias()
      );
    }
}

int NeuronLayer::size() {
  return this->neurons.size();
}

int NeuronLayer::parentSize() {
  return this->parent->size();
}

void NeuronLayer::activate(std::function<float(float)> activation_function) {
  if (this->parent == nullptr) return;
  for (Neuron& N : this->neurons) {
    N.activate(activation_function);
  }
}

std::vector<Neuron> *NeuronLayer::getNeuronsPtr() {
  return &this->neurons;
}

NeuronLayer *NeuronLayer::getParentPtr() {
  return this->parent;
}
