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

  void setWeight(float n, int index) {
    this->parent_weights[index] = n;
  }

  void setBias(float n) {
    this->bias = n;
  }

  float getBias() {return this->bias;}
  
  // only to be used in input layer
  void setActivation(float i) {
    this->activation = i;
  }
};

class NeuronLayer {
public:
  int size() {return this->neurons.size();}
  int parentSize() {return this->parent->size();}
  std::vector<Neuron> *getNeuronsPtr() {return &this->neurons;}
  NeuronLayer *getParentPtr() {return this->parent;}
  NeuronLayer(int size) {
    this->neurons = std::vector<Neuron>(size);
    this->parent = nullptr;
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

std::vector<float> backPropogate(NeuronLayer *L, std::vector<float> desired_activations) {
  std::vector<float> next_desired_activations = std::vector<float>(L->parentSize());
  std::vector<std::vector<float>> d_activation_matrix(L->size(), std::vector<float>(L->parentSize()));
  std::vector<Neuron> *parents = L->getParentPtr()->getNeuronsPtr();
  if (parents == nullptr) {
    return std::vector<float>(0);
  }
  //std::vector<std::vector<float>> d_activation_matrix(L->size(), std::vector<float>(L->parentSize()));
  for (int i_neuron = 0; i_neuron < L->size(); i_neuron ++) {
    Neuron &n = L->getNeuronsPtr()->at(i_neuron);
    for (int i_weight = 0; i_weight < parents->size(); i_weight ++) {
      float activation_difference = parents->at(i_weight).getActivation() - n.getActivation();
      n.setWeight(fastSigmoid(activation_difference), i_weight);
      d_activation_matrix[i_neuron][i_weight] = 0 - activation_difference;
    }
    n.activate();
    n.setBias(fastSigmoid(n.getActivation() - parents->at(i_).getActivation()));
  }
  for (int i = 0; i < d_activation_matrix.size(); i ++) {
    float activation = 0;
    for (float value : d_activation_matrix[i]) {
      activation += value;
    }
    activation /= d_activation_matrix[i].size();
    next_desired_activations[i] = activation;
  }
  return backPropogate(L->getParentPtr(), desired_activations);
}

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
  NeuronLayer input_layer(28*28);
  NeuronLayer layer1(28*28, &input_layer);
  NeuronLayer layer2(16, &layer1);
  NeuronLayer output(10, &layer2);
  while (getline(csv, line)) {
    int digit = setFromLine(input_layer.getNeuronsPtr(), line);
    input_layer.activate();
    layer1.activate();
    layer2.activate();
    output.activate();
    std::vector<float> desired_activations = std::vector<float>(10);
    desired_activations[digit] = 1;
    backPropogate(&output, desired_activations);
  };
  //printLayer(*layer1.getNeuronsPtr());
  //printLayer(*layer2.getNeuronsPtr(), 4);
  //printLayer(*output.getNeuronsPtr(), 10);
  
  return 0;
}
