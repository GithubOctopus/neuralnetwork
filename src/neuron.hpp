#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include <functional>

namespace NN {

float sigmoidFunction(float sum);
float sigmoidDerivativeFunction(float sum);

class Neuron {
private:
  std::vector<Neuron> *parents;
  std::vector<float> parent_weights;
  float bias;
  float activation;
  float sum;
public:
  Neuron() {}
  Neuron(
    std::vector<Neuron> *parents,
    std::vector<float> parent_weights,
    float bias
  ); 
  float getActivation();
  float activate(std::function<float(float)> activation_function);
  void setWeight(float n, int index);
  float getWeight(int i);
  float getSum();
  void setBias(float n);
  float getBias(); 
  void setActivation(float i);
};


}
#endif
