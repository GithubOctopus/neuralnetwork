#include <iostream>
#include <vector>

class Neuron{
private:
  int number_of_parents;
  std::vector<Neuron*> parents;
  std::vector<float> parent_weights;
  float bias;
  float activation;
public:
  Neuron(
    int number_of_parents,
    std::vector<Neuron*> parents,
    std::vector<float> parent_weights,
    float bias
  ) {
    this->bias = bias;
    this->number_of_parents = number_of_parents;
    this->parents = parents;
    this->parent_weights = parent_weights;
  }

};

int main() {
  std::cout << "hello world\n";
  return 0;
}
