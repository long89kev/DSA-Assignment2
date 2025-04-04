/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.cc to edit this template
 */

/* 
 * File:   Softmax.cpp
 * Author: ltsach
 * 
 * Created on August 25, 2024, 2:46 PM
 */

#include "layer/Softmax.h"
#include "ann/functions.h"
#include "sformat/fmt_lib.h"
#include <filesystem> //require C++17
namespace fs = std::filesystem;

Softmax::Softmax(int axis, string name): m_nAxis(axis) {
    if(trim(name).size() != 0) m_sName = name;
    else m_sName = "Softmax_" + to_string(++m_unLayer_idx);
}

Softmax::Softmax(const Softmax& orig) {
}

Softmax::~Softmax() {
}

xt::xarray<double> Softmax::forward(xt::xarray<double> X) {
    //YOUR CODE IS HERE
    m_aCached_Y = softmax(X, m_nAxis);
    return m_aCached_Y;
}
xt::xarray<double> Softmax::backward(xt::xarray<double> DY) {
    xt::xarray<double> y = m_aCached_Y;
    xt::xarray<double> DX = xt::zeros_like(DY);

    for (size_t i = 0; i < y.shape()[0]; ++i) {
        xt::xarray<double> yi = xt::view(y, i);
        xt::xarray<double> dyi = xt::view(DY, i);

        xt::xarray<double> diag_yi = xt::diag(yi);
        xt::xarray<double> outer_yi = xt::linalg::outer(yi, yi);

        xt::view(DX, i) = xt::linalg::dot(diag_yi - outer_yi, dyi);
    }

    return DX;
}
string Softmax::get_desc(){
    string desc = fmt::format("{:<10s}, {:<15s}: {:4d}",
                    "Softmax", this->getname(), m_nAxis);
    return desc;
}
