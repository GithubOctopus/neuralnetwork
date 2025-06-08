#include <cstdlib>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>

float activationFunction(float x) {
  return x / (2.0f * (1.0f + std::abs(x))) + 0.5f;
}

float activationFunctionDerivative(float x) {
  float denom = 1.0f + std::abs(x);
  return 1.0f / (2.0f * denom * denom);
}

std::vector<float> generateRandomWeights(int size) {
  std::vector<float> result(size);
  for (int i = 0; i < size; i ++) {
    float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 0.5f;
    result[i] =  r * 2 - 1;
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
    this->activation = activationFunction(this->getSum());
    return this->activation;
  };

  void setWeight(float n, int index) {
    this->parent_weights[index] = n;
  }

  float getWeight(int i) {
    return this->parent_weights[i];
  }

  float getSum() {
    float sum = 0;
    for (int i = 0; i < this->parents->size(); i ++) {
      sum += this->parents->at(i).getActivation() * this->parent_weights.at(i);
    }
    sum += this->bias;
    return sum;
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
    if (this->parent == nullptr) return;
    for (Neuron& N : this->neurons) {
      N.activate();
    }
  }
private:
  std::vector<Neuron> neurons;
  NeuronLayer *parent;
};

std::vector<float> backPropogate(NeuronLayer *L, std::vector<float> desired_activations) {
  NeuronLayer* this_layer = L;
  NeuronLayer* parent_layer = L->getParentPtr();
  if (parent_layer == nullptr) {return std::vector<float>(0);}
  std::vector<float> this_desired_activations = desired_activations;
  std::vector<std::vector<float>> parents_desired_activation_matrix(parent_layer->size(), std::vector<float>(parent_layer->size()));

  float learning_rate = 0.2f;

  // for each neuron in this layer
  for (int i = 0; i < this_layer->size(); i ++) {
    Neuron &this_neuron = this_layer->getNeuronsPtr()->at(i);

    float error = this_neuron.getActivation() - desired_activations[i];
    float activation_derivative = activationFunctionDerivative(this_neuron.getSum());

    // for each neuron in the parent layer
    for (int j = 0; j < parent_layer->size(); j ++) {
      Neuron &this_parent = parent_layer->getNeuronsPtr()->at(j);
      float old_weight = this_neuron.getWeight(j);
      float new_weight = old_weight - learning_rate * activation_derivative * error * this_parent.getActivation();
      this_neuron.setWeight(new_weight, j);
      float old_bias = this_neuron.getBias();
      float new_bias = old_bias - learning_rate * activation_derivative * error;
      this_neuron.setBias(new_bias);
      float ideal_parent_activation = this_parent.getActivation() - learning_rate * activation_derivative * error * new_weight;
    }
  }

  std::vector<float> parents_desired_activation(parent_layer->size());
  for (int i = 0; i < parents_desired_activation.size(); i ++) {
    float sum = 0;
    for (float f : parents_desired_activation_matrix[i]) {
      sum += f;
    }
    parents_desired_activation[i] = activationFunction(sum);
  }

  return backPropogate(parent_layer, parents_desired_activation);
}

void printLayer(std::vector<Neuron>& neurons, int width = 28) {
  int height = neurons.size() / width;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      float a = neurons[i * width + j].getActivation();
      std::cout << (a > 0 ? "1" : " ");
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
  //printLayer(*neurons);
  return digit;
}

int getAnswer(NeuronLayer *output) {
  int max_index = 0;
  float last_max = -std::numeric_limits<float>::infinity();  // ✅ fix here
  for (int i = 0; i < output->getNeuronsPtr()->size(); i++) {
    float act = output->getNeuronsPtr()->at(i).getActivation();
    if (act > last_max) {
      max_index = i;
      last_max = act;
    }
  }
  return max_index;
}


int main(int argc, char** argv) {
  for (int i = 0; i < argc; i ++) {
    std::cout << "argv[i] " << argv[i] << std::endl;
  }
  if (argc < 2) {
    return 1;
  }
  std::string file_path = argv[1];
  std::string line;
  NeuronLayer input_layer(28*28);
  NeuronLayer layer1(256, &input_layer);
  NeuronLayer layer2(128, &layer1);
  NeuronLayer output(10, &layer2);

  std::vector<bool> last_answers;
  int i = 0;
  for (int epoch = 0; epoch <= 3; epoch ++) {
    std::ifstream csv;
    csv.open(file_path, std::ios::in);

  
  while (getline(csv, line)) {
    int digit = setFromLine(input_layer.getNeuronsPtr(), line);
    i ++;
    layer1.activate();
    layer2.activate();
    output.activate();
    std::vector<float> desired_activations = std::vector<float>(10, 0.1f);
    desired_activations[digit] = 0.9f;
    if (getAnswer(&output) == digit) {
      //std::cout << "true" << std::endl;
      last_answers.push_back(true);
    } else {
      //std::cout << "false" << std::endl;
      last_answers.push_back(false);
    }
    if (last_answers.size() > 1000) {
      last_answers.erase(last_answers.begin());
    }
    if (i % 10 == 1) {
      int last_good = 0; for (bool b : last_answers) if (b) last_good ++;
      std::cout << "ratio: " << last_good << ":" << 1000 - last_good << std::endl;
      std::cout << "guess: " << getAnswer(&output) << std::endl;
      printLayer(*input_layer.getNeuronsPtr());
      std::cout << "Output activations: ";
      for (int i = 0; i < output.size(); ++i) {
        std::cout << output.getNeuronsPtr()->at(i).getActivation() << " ";
      }
      std::cout << "\n";
      i = 0; 
    }
    backPropogate(&output, desired_activations);

  };
  }
  //printLayer(*layer1.getNeuronsPtr());
  //printLayer(*layer2.getNeuronsPtr(), 4);
  //printLayer(*output.getNeuronsPtr(), 10);
  
  return 0;
}
