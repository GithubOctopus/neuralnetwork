#include <cstdlib>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>
#include <fstream>

float fastSigmoid(float x) {
  // f(x) = \frac{x}{2(1+|x|)} + \frac12
  return x / (2 * (1 + std::abs(x))) + 1./2.;
}


class Neuron{
private:
  int number_of_parents;
  std::vector<Neuron*> parents;
  std::vector<float> parent_weights;
  float bias;
  float activation;
public:
  Neuron() {
    number_of_parents = 0;
    bias = 0;
  }
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
  
  float getActivation() {
    return this->activation;
  }

  float activate() {
    float sum = 0;
    for (int i = 0; i < this->parents.size(); i ++) {
      sum += this->parents[i]->activation * this->parent_weights[i];
    }
    sum += this->bias;
    return fastSigmoid(sum);
  };
  
  // only to be used in input layer
  void setActivation(float i) {
    this->activation = i;
  }
};

void printLayer(std::vector<Neuron> &neurons) {
  for (int i = 0; i < 28; i ++) {
    for (int j = 0; j < 28; j ++) {
      float a = neurons[i*28+j].getActivation();
      if (a > 0.5) std::cout << "1 ";
      else std::cout << "0 ";
    }
    std::cout << "\n";
  }
}

int setFromLine(std::vector<Neuron> *neurons, std::string line) {
  int digit = std::stoi(line.substr(0, line.find(",")));
  int i = line.find(",") + 1;
  for (Neuron &N : *neurons) {
    std::string tok = line.substr(i, line.find(",", i) - i);
    i = line.find(",", i) + 1;
    unsigned char activation_char = std::stoi(tok);

    float activation_float = (float)activation_char / 255;
    std::cout << activation_float << std::endl;
    N.setActivation(activation_float);
  }
  printLayer(*neurons);
  return digit;
}

int main(int argc, char** argv) {
  for (int i = 0; i < argc; i ++) {
    std::cout << "argv[i] " << argv[i] << std::endl;
  }
  if (argc < 1) {
    return 1;
  }
  std::string file_path = argv[1];
  std::ifstream csv;
  csv.open(file_path, std::ios::in);
  std::string line;
  getline(csv, line);
  std::vector<Neuron> input_layer(28*28);
  int digit = setFromLine(&input_layer, line);
  std::cout << "digit";
  
  return 0;
}
