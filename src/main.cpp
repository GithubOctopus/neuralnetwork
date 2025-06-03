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

std::vector<float> generateRandomWeights(int size) {
  std::vector<float> result;
  for (int i = 0; i < size; i ++) {
    float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    result.push_back(r);
  }
  return result;
}

float generateRandomBias() {
  return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}


class Neuron{
private:
  std::vector<Neuron> *parents;
  std::vector<float> parent_weights;
  float bias;
  float activation;
public:
  Neuron() {
  }
  Neuron(
    std::vector<Neuron> *parents,
    std::vector<float> parent_weights,
    float bias
  ) {
    this->bias = bias;
    this->parents = parents;
    this->parent_weights = parent_weights;
  }
  
  float getActivation() {
    return this->activation;
  }

  float activate() {
    float sum = 0;
    for (int i = 0; i < this->parents->size(); i ++) {
      sum += this->parents->at(i).getActivation() * this->parent_weights.at(i);
    }
    sum += this->bias;
    return fastSigmoid(sum);
  };
  
  // only to be used in input layer
  void setActivation(float i) {
    this->activation = i;
  }
};

class NeuronLayer {
  int size() {return this->neurons.size();}
  std::vector<Neuron> *getNeuronsPtr() {return &this->neurons;}
  NeuronLayer(int size) {
    this->neurons = std::vector<Neuron>(size);
  }
  NeuronLayer(int size, NeuronLayer* parent) {
    this->parent = parent;
    this->neurons = std::vector<Neuron>(size);
    for (int i = 0; i < size; i ++) {
      std::vector<float> weights = generateRandomWeights(parent->size());
      float bias = generateRandomBias();
      this->neurons[i] = Neuron(
        parent->getNeuronsPtr(), 
        weights,
        bias
      );
    }
  };
  private:
  std::vector<Neuron> neurons;
  NeuronLayer *parent;
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
  std::cout << "digit" << digit << "\n";
  
  return 0;
}
