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
  std::vector<float> result(size);
  for (int i = 0; i < size; i ++) {
    float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    result[i] = r * 2 - 1;
  }
  return result;
}

float generateRandomBias() {
  float i = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  return i * 2 - 1;
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
    this->activation = fastSigmoid(sum);
    return this->activation;
  };
  
  // only to be used in input layer
  void setActivation(float i) {
    this->activation = i;
  }
};

class NeuronLayer {
public:
  int size() {return this->neurons.size();}
  std::vector<Neuron> *getNeuronsPtr() {return &this->neurons;}
  NeuronLayer(int size) {
    this->neurons = std::vector<Neuron>(size);
  }
  NeuronLayer(int size, NeuronLayer* parent) {
    this->parent = parent;
    this->neurons = std::vector<Neuron>(size);
    for (int i = 0; i < size; i ++) {
      this->neurons[i] = Neuron(
        parent->getNeuronsPtr(), 
        generateRandomWeights(parent->size()),
        generateRandomBias()
      );
    }
  }
  void activate() {
    for (Neuron& N : this->neurons) {
      N.activate();
    }
  }
private:
  std::vector<Neuron> neurons;
  NeuronLayer *parent;
};

void printLayer(std::vector<Neuron>& neurons, int width = 28) {
  int height = neurons.size() / width;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      float a = neurons[i * width + j].getActivation();
      std::cout << (a > 0.5 ? "1 " : "0 ");
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
  NeuronLayer input_layer(28*28);
  NeuronLayer layer1(28*28, &input_layer);
  NeuronLayer layer2(16, &layer1);
  NeuronLayer output(10, &layer2);
  setFromLine(input_layer.getNeuronsPtr(), line);
  layer1.activate();
  layer2.activate();
  output.activate();
  printLayer(*layer1.getNeuronsPtr());
  printLayer(*layer2.getNeuronsPtr(), 4);
  printLayer(*output.getNeuronsPtr(), 10);
  
  return 0;
}
