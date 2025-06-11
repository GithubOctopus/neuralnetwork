#include <iostream>
#include <ostream>
#include <string>
#include <vector>
#include <fstream>

#include "neuron.hpp"
#include "neuralnetwork.hpp"


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

std::pair<std::vector<float>, int> getFromLine(std::string line, int size) {
  int digit = std::stoi(line.substr(0, line.find(",")));
  int i = line.find(",") + 1;
  std::vector<float> activations;
  for (int _ = 0; _ < size; _ ++) {
    std::string tok = line.substr(i, line.find(",", i) - i);
    i = line.find(",", i) + 1;
    unsigned char activation_char = std::stoi(tok);

    float activation_float = (float)activation_char / 255;
    activations.push_back(activation_float);
  }
  std::pair<std::vector<float>, int> result(activations, digit);
  return result;
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

  int num_layers = 4;
  int layer_sizes[] = {28*28, 256, 128, 10};
  float learning_rate = 0.1f;

  NN::NeuralNetwork network(num_layers, layer_sizes, learning_rate);


  std::vector<bool> last_answers;
  int i = 0;
  for (int epoch = 0; epoch <= 3; epoch ++) {
    std::ifstream csv;
    csv.open(file_path, std::ios::in);

  
    while (getline(csv, line)) {
      i ++;
      auto line_data = getFromLine(line, 28*28);
      std::vector<float> ideal_activations = std::vector<float>(10, 0.1f);
      int digit = line_data.second;
      std::cout << "expected: " << digit << " ";
      ideal_activations[digit] = 0.9f;
      auto acts = network.train(line_data.first, ideal_activations);
      std::cout << "got: " << network.result() << std::endl;
      for (auto a : acts) {
        std::cout << a << ", ";
      }
      std::cout << std::endl;


    };
  }
  
  return 0;
}
