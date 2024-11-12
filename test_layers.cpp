#include <iostream>
#include <cassert>
#include "include/ann/layer/Sigmoid.h"
#include "include/ann/layer/ReLU.h"
#include "include/ann/loss/CrossEntropy.h"
#include "include/ann/layer/ILayer.h"
#include "include/ann/layer/FCLayer.h"
#include "include/tensor/xtensor_lib.h"
#include "include/ann/layer/Tanh.h"
#include "include/ann/layer/Softmax.h"

void test_sigmoid_layer() {
    Sigmoid sigmoid("SigmoidTest");
    xt::xarray<double> input = {-1.0, 0.0, 1.0};
    xt::xarray<double> expected_output = 1.0 / (1.0 + xt::exp(-input));

    xt::xarray<double> output = sigmoid.forward(input);
    assert(xt::allclose(output, expected_output, 1e-5));

    xt::xarray<double> grad_output = xt::ones_like(output);
    xt::xarray<double> grad_input = sigmoid.backward(grad_output);
    xt::xarray<double> expected_grad_input = grad_output * output * (1.0 - output);
    assert(xt::allclose(grad_input, expected_grad_input, 1e-5));

    std::cout << "Sigmoid layer test passed." << std::endl;
}

void test_relu_layer() {
    ReLU relu("ReLUTest");
    xt::xarray<double> input = {-1.0, 0.0, 1.0};
    xt::xarray<double> expected_output = xt::maximum(0.0, input);

    xt::xarray<double> output = relu.forward(input);
    assert(xt::allclose(output, expected_output));

    xt::xarray<double> grad_output = xt::ones_like(output);
    xt::xarray<double> grad_input = relu.backward(grad_output);
    xt::xarray<double> expected_grad_input = grad_output * (input > 0.0);
    assert(xt::allclose(grad_input, expected_grad_input));

    std::cout << "ReLU layer test passed." << std::endl;
}

void test_cross_entropy_loss() {
    CrossEntropy loss_layer;
    xt::xarray<double> predictions = {{0.8, 0.2}, {0.1, 0.9}};
    xt::xarray<double> targets = {{1, 0}, {0, 1}};

    double loss = loss_layer.forward(predictions, targets);
    double expected_loss = -xt::mean(xt::sum(targets * xt::log(predictions), {1}))(0);
    assert(std::abs(loss - expected_loss) < 1e-5);

    xt::xarray<double> grad_input = loss_layer.backward();
    xt::xarray<double> expected_grad_input = (predictions - targets) / predictions.shape()[0];
    assert(xt::allclose(grad_input, expected_grad_input, 1e-5));

    std::cout << "CrossEntropy loss test passed." << std::endl;
}

void test_tanh_layer() {
    Tanh tanh_layer("TanhTest");
    xt::xarray<double> input = {-1.0, 0.0, 1.0};
    xt::xarray<double> expected_output = xt::tanh(input);

    xt::xarray<double> output = tanh_layer.forward(input);
    assert(xt::allclose(output, expected_output, 1e-5));

    xt::xarray<double> grad_output = xt::ones_like(output);
    xt::xarray<double> grad_input = tanh_layer.backward(grad_output);
    xt::xarray<double> expected_grad_input = grad_output * (1.0 - xt::pow(output, 2));
    assert(xt::allclose(grad_input, expected_grad_input, 1e-5));

    std::cout << "Tanh layer test passed." << std::endl;
}

void test_softmax_layer() {
    Softmax softmax_layer(-1, "SoftmaxTest");
    xt::xarray<double> input = {1.0, 2.0, 3.0};
    xt::xarray<double> shifted_input = input - xt::amax(input);
    xt::xarray<double> exp_input = xt::exp(shifted_input);
    xt::xarray<double> expected_output = exp_input / xt::sum(exp_input);

    xt::xarray<double> output = softmax_layer.forward(input);
    assert(xt::allclose(output, expected_output, 1e-5));

    std::cout << "Softmax layer test passed." << std::endl;
}

void test_fc_layer() {
    FCLayer fc_layer(3, 2, true);
    
    double_tensor weights = xt::xarray<double>::from_shape({2, 3}); 
    weights(0,0) = 0.1; weights(0,1) = 0.2; weights(0,2) = 0.3;
    weights(1,0) = 0.4; weights(1,1) = 0.5; weights(1,2) = 0.6;
    fc_layer.set_weights(weights);
    
    double_tensor bias = xt::xarray<double>::from_shape({2}); 
    bias(0) = 0.1; bias(1) = 0.2;
    fc_layer.set_bias(bias);

    double_tensor input = xt::xarray<double>::from_shape({3}); 
    input(0) = 1.0; input(1) = 2.0; input(2) = 3.0;
    
    double_tensor expected_output = xt::linalg::dot(weights, input) + bias;
    double_tensor output = fc_layer.forward(input);
    
    assert(xt::allclose(output, expected_output, 1e-5));

    double_tensor grad_output = xt::xarray<double>::from_shape({2}); 
    grad_output(0) = 0.1; grad_output(1) = 0.2;
    
    double_tensor expected_grad_input = xt::linalg::dot(xt::transpose(weights), grad_output);
    double_tensor grad_input = fc_layer.backward(grad_output);
    
    assert(xt::allclose(grad_input, expected_grad_input, 1e-5));

    std::cout << "Fully Connected layer test passed." << std::endl;
}

int main() {
    test_sigmoid_layer();
    test_relu_layer();
    test_tanh_layer();
    test_softmax_layer();
    test_fc_layer();
    test_cross_entropy_loss();
    return 0;
}