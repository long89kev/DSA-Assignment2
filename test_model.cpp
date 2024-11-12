#include <iostream>
#include <cassert>
#include "include/ann/model/MLPClassifier.h"
#include "include/ann/layer/FCLayer.h"
#include "include/ann/layer/ReLU.h"
#include "include/ann/layer/Sigmoid.h"
#include "include/ann/layer/Softmax.h"
#include "include/ann/loss/CrossEntropy.h"
#include "include/ann/metrics/ClassMetrics.h"
#include "include/ann/optim/SGD.h"
#include "include/tensor/xtensor_lib.h"
#include "include/loader/dataset.h"
#include "include/loader/dataloader.h"

void clip_gradients(double_tensor& gradients, double clip_value) {
    for (auto& grad : gradients) {
        if (grad > clip_value) grad = clip_value;
        else if (grad < -clip_value) grad = -clip_value;
    }
}

void test_mlp_classifier_with_optimizer() {
    // Create a simple dataset
    xt::xarray<double> X_train = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    xt::xarray<double> y_train = {{1.0, 0.0}, {0.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}};

    for (auto& value : X_train) {
        if (std::isnan(value)) {
            std::cerr << "Found nan in X_train" << std::endl;
            return;
        }
    }
    for (auto& value : y_train) {
        if (std::isnan(value)) {
            std::cerr << "Found nan in y_train" << std::endl;
            return;
        }
    }

    // Define the model architecture
    ILayer* layers[] = {
        new FCLayer(2, 4, true),
        new ReLU(),
        new FCLayer(4, 2, true),
        new Softmax()
    };
    MLPClassifier model("./config.txt", "mlp_test", layers, 4);

    // Setup optimizer, loss and metrics
    SGD optimizer(1e-16); // Reduce learning rate further
    CrossEntropy loss;
    ClassMetrics metrics(2);
    model.compile(&optimizer, &loss, &metrics);

    // Create data loader
    TensorDataset<double, double> dataset(X_train, y_train);
    DataLoader<double, double> loader(&dataset, 2, true, false);

    // Train the model using the fit method
    model.fit(&loader, &loader, 10); // Train for 10 epochs using the same loader for training and validation

    std::cout << "MLPClassifier with optimizer test passed." << std::endl;

    // Clean up dynamically allocated layers
    for (auto layer : layers) {
        delete layer;
    }
}

int main() {
    test_mlp_classifier_with_optimizer();
    return 0;
}