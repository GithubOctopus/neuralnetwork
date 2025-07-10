#include "neuralnetwork.hpp"
#include "neuronlayer.hpp"

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
  learning_rate(learning_rate)
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
