#include "neuron.hpp"
#include <cmath>

using namespace NN;

float NN::sigmoidFunction(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

float NN::sigmoidDerivativeFunction(float x) {
  float fx = NN::sigmoidFunction(x);
  return fx * (1.0f - fx);
}

Neuron::Neuron(
  std::vector<Neuron> *parents,
  std::vector<float> parent_weights,
  float bias
) : 
  parents(parents), 
  parent_weights(std::move(parent_weights)), 
  bias(bias) 
{}


float Neuron::activate(std::function<float(float)> activation_function) {
  float sum = 0;
  for (int i = 0; i < this->parents->size(); i ++) {
    sum += this->parents->at(i).getActivation() * this->parent_weights.at(i);
  }
  sum += this->bias;
  this->sum = sum;
  this->activation = activation_function(sum);
  return this->activation;
}


float Neuron::getSum() {
  return this->sum;
}
float Neuron::getBias() {
  return this->bias;
}
float Neuron::getActivation() {
  return this->activation;
}
float Neuron::getWeight(int i) {
  return this->parent_weights[i];
}

void Neuron::setBias(float bias) {
  this->bias = bias;
}
void Neuron::setActivation(float activation) {
  this->activation = activation;
}

void Neuron::setWeight(float n, int index) {
  this->parent_weights[index] = n;
}
