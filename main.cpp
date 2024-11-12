#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
using namespace std;

#include <filesystem> //require C++17
namespace fs = std::filesystem;

#include "include/list/listheader.h"

// #include "sformat/fmt_lib.h"
// #include "tensor/xtensor_lib.h"
// #include "ann/annheader.h"
// #include "loader/dataset.h"
// #include "loader/dataloader.h"
// #include "config/Config.h"
// #include "dataset/DSFactory.h"
// #include "optim/Adagrad.h"
// #include "optim/Adam.h"
// #include "modelzoo/twoclasses.h"
// #include "modelzoo/threeclasses.h"

#include "demo/heap/HeapDemo.h"
#include "demo/hash/xMapDemo.h"


int main(int argc, char** argv) {
    //dataloader:
    //case_data_wo_label_1();
    //case_data_wi_label_1();
    //case_batch_larger_nsamples();
    
    //Classification:
    // twoclasses_classification();
    //threeclasses_classification();

    // cout << "Heap Demo: " << endl;
    // heapDemo1();
    // heapDemo2();
    // heapDemo3();

    cout << "Hash Demo: " << endl;
    cout << "Hash Demo 1: " << endl;
    hashDemo1();
    cout << "Hash Demo 2: " << endl;
    hashDemo2();
    cout << "Hash Demo 3: " << endl;
    hashDemo3();
    cout << "Hash Demo 4: " << endl;
    hashDemo4();
    cout << "Hash Demo 5: " << endl;
    hashDemo5();
    cout << "Hash Demo 6: " << endl;
    hashDemo6();
    cout << "Hash Demo 7: " << endl;
    hashDemo7();
    cout << "Country Demo: " << endl;
    countryDemo();
 
    return 0;
}
