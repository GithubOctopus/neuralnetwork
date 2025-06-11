#include <iostream>
#include <ostream>
#include <string>
#include <vector>
#include <fstream>

#include "neuron.hpp"
#include "neuronlayer.hpp"


void backPropagateHelper(NN::NeuronLayer* this_layer, std::vector<float> deltas) {
  float learning_rate = 0.1f;
  NN::NeuronLayer *parent_layer = this_layer->getParentPtr();
  if (parent_layer == nullptr) {return;}
  std::vector<float> parent_deltas(parent_layer->size(), 0.0f);
  // for each neuron in layer
  for (int i = 0; i < this_layer->size(); i++) {
    NN::Neuron& this_neuron = this_layer->getNeuronsPtr()->at(i);
    float delta = deltas[i];

    // Bias update
    this_neuron.setBias(this_neuron.getBias() - learning_rate * delta);

    // Weights update
    for (int j = 0; j < parent_layer->size(); j++) {

      NN::Neuron &this_parent = parent_layer->getNeuronsPtr()->at(j);
      float old_weight = this_neuron.getWeight(j);
      float new_weight = old_weight - learning_rate * delta * this_parent.getActivation();
      this_neuron.setWeight(new_weight, j);
    }
  }

  if (parent_layer->getParentPtr() == nullptr) {
    return;
  }

  for (int j = 0; j < parent_layer->size(); j++) {
    float sum = 0.0f;
    for (int i = 0; i < this_layer->size(); i++) {
      float weight = this_layer->getNeuronsPtr()->at(i).getWeight(j);
      sum += deltas[i] * weight;
    }
    float parent_sum = parent_layer->getNeuronsPtr()->at(j).getSum();
    parent_deltas[j] = sum * NN::sigmoidDerivativeFunction(parent_sum);
  }
  backPropagateHelper(parent_layer, parent_deltas);
};

void backPropagate(NN::NeuronLayer *this_layer, std::vector<float> target_activations) {
  std::vector<float> deltas(this_layer->size());

  for (int i = 0; i < this_layer->size(); i ++) {

    NN::Neuron &this_neuron = this_layer->getNeuronsPtr()->at(i);
    float error = this_neuron.getActivation() - target_activations[i];
    float activation_derivative = NN::sigmoidDerivativeFunction(this_neuron.getSum());
    float delta = error * activation_derivative;
    deltas[i] = delta;

    std::cout << "delta " << delta << std::endl;
    std::cout << "error: " << error << std::endl;
    std::cout << "activation_derivative = " << activation_derivative << std::endl;
  }
  backPropagateHelper(this_layer, deltas);
};

void printLayer(std::vector<NN::Neuron>& neurons, int width = 28) {
  int height = neurons.size() / width;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      float a = neurons[i * width + j].getActivation();
      std::cout << (a > 0 ? "1" : " ");
    }
    std::cout << "\n";
  }
}

int setFromLine(std::vector<NN::Neuron> *neurons, std::string line) {
  int digit = std::stoi(line.substr(0, line.find(",")));
  int i = line.find(",") + 1;
  for (NN::Neuron &N : *neurons) {
    std::string tok = line.substr(i, line.find(",", i) - i);
    i = line.find(",", i) + 1;
    unsigned char activation_char = std::stoi(tok);

    float activation_float = (float)activation_char / 255;
    N.setActivation(activation_float);
  }
  return digit;
}

int getAnswer(NN::NeuronLayer *output) {
  int max_index = 0;
  float last_max = -100.0f;  // ✅ fix here
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
  NN::NeuronLayer input_layer(28*28);
  NN::NeuronLayer layer1(256, &input_layer);
  NN::NeuronLayer layer2(128, &layer1);
  NN::NeuronLayer output(10, &layer2);

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
    backPropagate(&output, desired_activations);

  };
  }
  
  return 0;
}
