#include "neuralnetwork.hpp"
#include "neuronlayer.hpp"
#include <streambuf>
#include <string>
#include <sstream>
#include <iostream>
using namespace NN;


void backPropagateHidden(
  NeuronLayer *this_layer,
  std::vector<float> deltas,
  const std::function<float(float)> &derivative_function,
  float learning_rate
) {

  NeuronLayer *parent_layer = this_layer->getParentPtr();
  if (parent_layer == nullptr) {return;}
  std::vector<float> parent_deltas(parent_layer->size(), 0.0f);
  // for each neuron in layer
  for (int i = 0; i < this_layer->size(); i++) {
    Neuron& this_neuron = this_layer->getNeuronsPtr()->at(i);
    float delta = deltas[i];

    // bias update
    this_neuron.setBias(this_neuron.getBias() - learning_rate * delta);

    // weights update
    for (int j = 0; j < parent_layer->size(); j++) {

      Neuron &this_parent = parent_layer->getNeuronsPtr()->at(j);
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
    parent_deltas[j] = sum * derivative_function(parent_sum);
  }
  backPropagateHidden(parent_layer, parent_deltas, derivative_function, learning_rate);
}


void backPropagateOutput(
  NeuronLayer *output_layer,
  std::vector<float> ideal_activations,
  const std::function<float(float)> &derivative_function,
  float learning_rate
) {
  std::vector<float> deltas(output_layer->size());
  for (int i = 0; i < output_layer->size(); i ++) {
    Neuron &this_neuron = output_layer->getNeuronsPtr()->at(i);
    float error = this_neuron.getActivation() - ideal_activations[i];
    float activation_derivative = derivative_function(this_neuron.getSum());
    float delta = error * activation_derivative;
    deltas[i] = delta;
  }
  backPropagateHidden(output_layer, deltas, derivative_function, learning_rate);


}


std::vector<float> NeuralNetwork::run(std::vector<float> input) {
  NeuronLayer *input_layer = this->layers[0];
  NeuronLayer *output_layer = this->layers[this->layers.size()-1];
  if (input_layer->size() != input.size()) {
    return std::vector<float>(0);
  }
  std::vector<float> output(output_layer->size());
  for (int i = 0; i < input_layer->size(); i ++) {
    input_layer->getNeuronsPtr()->at(i).setActivation(input[i]);
  }
  for (NeuronLayer *L : this->layers) {
    L->activate(this->activation_function);
  }
  for (int i = 0; i < output_layer->size(); i ++) {
    output[i] = output_layer->getNeuronsPtr()->at(i).getActivation();
  }
  Neuron &n = output_layer->getNeuronsPtr()->at(0);

  return output;
}

std::vector<float> NeuralNetwork::train(
  std::vector<float> input,
  std::vector<float> ideal_output
) {
  NeuronLayer *input_layer = this->layers[0];
  NeuronLayer *output_layer = this->layers[this->layers.size()-1];
  this->run(input);

  backPropagateOutput(
    output_layer,
    ideal_output,
    this->activation_function_derivative,
    this->learning_rate
  );
  std::vector<float> actual_output(output_layer->size());
  for (int i = 0; i < output_layer->size(); i ++) {
    actual_output[i] = output_layer->getNeuronsPtr()->at(i).getActivation();

  }
  return actual_output;
}

int NeuralNetwork::result() {
  NeuronLayer *output_layer = this->layers[this->layers.size()-1];
  return output_layer->mostActivated().second;
}

NeuralNetwork::NeuralNetwork(
  int num_layers,
  int* layer_sizes,
  float learning_rate,
  const std::function<float(float)> &activation_function,
  const std::function<float(float)> &activation_function_derivative,
  const std::function<float()> &generate_bias,
  const std::function<std::vector<float>(int)> &generate_weights
) :
  activation_function_derivative(activation_function_derivative),
  activation_function(activation_function),
  learning_rate(learning_rate),
  generate_weights(generate_weights),
  generate_bias(generate_bias)
{
  this->layers = std::vector<NeuronLayer*>(num_layers);
  this->layers[0] = new NeuronLayer(layer_sizes[0]); // input layer
  for (int i = 1; i < num_layers; i ++) {
    layers[i] = new NeuronLayer(
      layer_sizes[i],
      layers[i-1],
      generate_bias,
      generate_weights
    );
  }
}

void NeuralNetwork::resetWeightsAndBiases() {
  this->layers = std::vector<NeuronLayer*>(this->layers.size());
  this->layers[0] = new NeuronLayer(this->layers[0]->size()); // input layer
  for (int i = 1; i < this->layers.size(); i ++) {
    layers[i] = new NeuronLayer(
      this->layers[i]->size(),
      layers[i-1],
      generate_bias,
      generate_weights
    );
  }
}

float NeuralNetwork::getLearningRate() {
  return this->learning_rate;
}

void NeuralNetwork::setLearningRate(float f) {
  this->learning_rate = f;
}

bool NeuralNetwork::writeToFile(std::ostream &o) const {
  for (NeuronLayer *l : this->layers) {
    o << l->size() << ",";
  }
  o << "\n";
  for (NeuronLayer *l : this->layers) {
    if (l->getParentPtr() != nullptr) {
      for (Neuron &n : *l->getNeuronsPtr()) {
        o << n.getBias() << ",";
        for (int w_i = 0; w_i < l->getParentPtr()->getNeuronsPtr()->size(); w_i ++) {
          o << n.getWeight(w_i) << ",";
        }
        o << "\n";
      }
    }
  }
  return true;
}

bool NeuralNetwork::readFromFile(std::istream &in) {
  std::string ln;
  if (!std::getline(in, ln)) return false;
  std::stringstream stream(ln);
  std::vector<int> layers_sizes;
  std::string size;
  while (std::getline(stream, size, ',')) {
    layers_sizes.push_back(std::stoi(size));
  }

  if (this->layers.size() != layers_sizes.size()) {
    return false;
  }

  for (int i = 0; i < this->layers.size(); i ++) {
    if (this->layers[i]->size() != layers_sizes[i]) {
      return false;
    }
  }
  for (int layer_index = 0; layer_index < layers_sizes.size(); layer_index ++) {

    auto this_layer = this->layers.at(layer_index);
    if (this_layer->getParentPtr() == nullptr) {continue;} // input layer
    int num_weights = this_layer->getParentPtr()->size();
    int num_neurons = this_layer->size();


    for (int neuron_index = 0; neuron_index < num_neurons; neuron_index ++) {
      std::cout << "Neuron " << layer_index << "|" << neuron_index << std::endl;
      Neuron &this_neuron = this_layer->getNeuronsPtr()->at(neuron_index);

      std::string line;
      if (!std::getline(in, line)) {
        return false; 
      }
      std::stringstream line_stream(line);
      std::string tok_str;
      std::getline(line_stream, tok_str, ',');
      float bias = std::stof(tok_str);
      std::cout << "bias: " << bias << std::endl;
      this_neuron.setBias(bias);


      for (int weight_index = 0; weight_index < num_weights; weight_index ++) {
        std::getline(line_stream, tok_str, ',');
        float weight = std::stof(tok_str);
        this_neuron.setWeight(weight, weight_index);
        std:: cout << "\tweight " << weight_index << ": " << weight << std::endl;
      }
    }
  }

  return true;
}
