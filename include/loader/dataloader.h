/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/* 
 * File:   dataloader.h
 * Author: ltsach
 *
 * Created on September 2, 2024, 4:01 PM
 */

#ifndef DATALOADER_H
#define DATALOADER_H
#include "tensor/xtensor_lib.h"
#include "loader/dataset.h"

using namespace std;

template<typename DType, typename LType>
class DataLoader{
public:
    class Iterator; //forward declaration for class Iterator
    int len(){ return ptr_dataset->len(); }
    
private:
    Dataset<DType, LType>* ptr_dataset;
    int batch_size;
    bool shuffle;
    bool drop_last;
    int nbatch;
    ulong_tensor item_indices;
    int m_seed;
    int index;
    
public:
    DataLoader(Dataset<DType, LType>* ptr_dataset, 
            int batch_size, bool shuffle=true, 
            bool drop_last=false, int seed=-1)
                : ptr_dataset(ptr_dataset), 
                batch_size(batch_size), 
                shuffle(shuffle),
                m_seed(seed), drop_last(drop_last){
            nbatch = ptr_dataset->len()/batch_size;
            item_indices = xt::arange(0, ptr_dataset->len());
            if(shuffle){
                if(m_seed >= 0) xt::random::seed(m_seed);
                xt::random::shuffle(item_indices);
            }
            index = 0;
    }
    virtual ~DataLoader(){}
    
    //New method: from V2: begin
    int get_batch_size(){ return batch_size; }
    int get_sample_count(){ return ptr_dataset->len(); }
    int get_total_batch(){return int(ptr_dataset->len()/batch_size); }
    
    //New method: from V2: end
    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// START: Section                                                     //
    /////////////////////////////////////////////////////////////////////////
public:
    Iterator begin(){
        //YOUR CODE IS HERE
        this->index = 0;
        if(this->batch_size > ptr_dataset->len()){
           return end();
        }
        return Iterator(this, 0);
    }
    Iterator end(){
        //YOUR CODE IS HERE
        int end_index = ptr_dataset->len();
        if(drop_last){
            end_index = ptr_dataset->len() - ptr_dataset->len() % batch_size;
        }
        bool end;
        return Iterator(this, end_index);
    }
    
    //BEGIN of Iterator

    //YOUR CODE IS HERE: to define iterator

    class Iterator{
    public:
        Iterator(DataLoader<DType, LType>* ptr_dataloader, int end_index = 0){
            this->ptr_dataloader = ptr_dataloader;
            this->current_index = 0;
            this->created = false;
            // if(this->ptr_dataloader->batch_size > ptr_dataloader->ptr_dataset->len()){
            //     this->current_index = ptr_dataloader->ptr_dataset->len();
            // }
            if(end_index == 0){
                this->current_index = ptr_dataloader->index;
            } else {
                this->current_index = end_index;
            }
        }
        
        Batch<DType, LType> operator*(){
            if(!created){
                current_batch = createBatch();
                created = true;
            }
            return current_batch;
        }

        Iterator& operator++(){
            if (current_index >= ptr_dataloader->ptr_dataset->len()) current_index = ptr_dataloader->ptr_dataset->len();
            created = false;
            return *this;
        }

        Iterator operator++(int){
            Iterator it = *this;
            ++(*this);
            return it;
        }
        
        bool operator!=(const Iterator& other) const{
            return current_index != other.current_index;
        }



    private:
        DataLoader<DType, LType>* ptr_dataloader;
        int current_index;
        Batch<DType, LType> current_batch = Batch<DType, LType>(xt::xarray<DType>(), xt::xarray<LType>());
        bool created = false;

    public:
        Batch<DType, LType> createBatch(){
            Dataset<DType, LType>* ptr_dataset = ptr_dataloader->ptr_dataset;
            xt::svector<unsigned long> data_shape = ptr_dataset->get_data_shape();
            int size_Batch = ptr_dataloader->batch_size;
            if(!ptr_dataloader->drop_last){
                // cout << "current_index: " << current_index << endl;
                if (current_index + ptr_dataloader->batch_size + (ptr_dataloader->len() % ptr_dataloader->batch_size) >= ptr_dataset->len()) {
                    // size_Batch = ptr_dataset->len() - current_index + ptr_dataloader->batch_size;
                    size_Batch = ptr_dataloader->batch_size + (ptr_dataloader->len() % ptr_dataloader->batch_size);
                }
                // cout << "size_Batch: " << size_Batch << endl;
                // cout << "current_index: " << current_index << endl;
            } else size_Batch = ptr_dataloader->batch_size;
            // if(ptr_dataloader->batch_size > ptr_dataset->len()) {
            //     return Batch<DType, LType>(xt::xarray<DType>(), xt::xarray<LType>());
            // }
            data_shape[0] = size_Batch;
            xt::xarray<DType> data = xt::empty<DType>(data_shape);
            bool is_label = ptr_dataset->get_label_shape().size() > 0;
            if(is_label){
                xt::svector<unsigned long> label_shape = ptr_dataset->get_label_shape();
                label_shape[0] = size_Batch;
                xt::xarray<LType> label = xt::empty<LType>(label_shape);
                for(int i = 0; i < size_Batch; i++){
                    DataLabel<DType, LType> dl = ptr_dataset->getitem(ptr_dataloader->item_indices[current_index]);
                    xt::view(data, i, xt::all(), xt::all()) = dl.getData();
                    xt::view(label, i, xt::all(), xt::all()) = dl.getLabel();
                    current_index++;
                }
                ptr_dataloader->index = current_index;
                return Batch<DType, LType>(data, label);
            } else {
                for(int i = 0; i < size_Batch; i++){
                    DataLabel<DType, LType> dl = ptr_dataset->getitem(ptr_dataloader->item_indices[current_index]);
                    xt::view(data, i, xt::all(), xt::all()) = dl.getData();
                    current_index++;
                }
                ptr_dataloader->index = current_index;
                return Batch<DType, LType>(data, xt::xarray<LType>());
            }
        }
    };   

    //END of Iterator
    
    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// END: Section                                                       //
    /////////////////////////////////////////////////////////////////////////
};


#endif /* DATALOADER_H */

