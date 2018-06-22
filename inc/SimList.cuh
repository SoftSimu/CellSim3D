#ifndef SimList_CUH
#define SIMLIST_CUH
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <string>
#include <exception>
#include "Types.cuh"
struct base_n{
    static size_t used_host_mem;
};

template<typename T>
struct SimList1D: base_n{
    size_t n;

    thrust::device_vector<T> d;
    thrust::host_vector<T> h;

    T* devPtr;
    T* hostPtr;

    // This is proper constructor delegation in C++11
    SimList1D(long int _n): SimList1D(_n, T(0)){}
    
    SimList1D(long int _n, T _val)
    try : n(_n), h(_n, _val), d(_n, _val){
        std::cout << " yay " << _n << std::endl;
        base_n::used_host_mem += n*sizeof(T);
        devPtr = thrust::raw_pointer_cast(&d[0]);
        hostPtr = thrust::raw_pointer_cast(&h[0]);
    } catch (const std::exception& e){
        std::cout << "Bad Memory Exception while trying to allocate "
                  << sizeof(T)*_n/(1024*1024) << " GB." << std::endl;
        std::cout << e.what() << std::endl; 
        throw e; 
    }

    ~SimList1D(){
        base_n::used_host_mem -= n*sizeof(T);
    }

    // copy constructor
    SimList1D(const SimList1D& other): n(other.n), d(other.n), h(other.n){
        thrust::copy(other.d.begin(), other.d.end(), d.begin());
        thrust::copy(other.h.begin(), other.h.end(), h.begin());
    }

    // copy assignment
    void operator=(const SimList1D& other){
        thrust::copy(other.d.begin(), other.d.end(), d.begin());
        thrust::copy(other.h.begin(), other.h.end(), h.begin());
    }

    void CopyToDevice(size_t _n, size_t offset){
        thrust::copy(h.begin()+offset, h.begin()+offset+_n, d.begin()+offset);
    }

    void CopyToDevice(){
        CopyToDevice(n, 0);
    }

    void CopyToHost(size_t _n, size_t offset){
        thrust::copy(d.begin()+offset, d.begin()+offset+_n, h.begin()+offset);
    }

    void CopyToHost(){
        CopyToHost(n, 0);
    }

    void Fill(T _val, size_t _n, size_t offset){
        thrust::fill(h.begin()+offset, h.begin()+offset+_n, _val);
        CopyToDevice(_n, offset);
    }

    void Fill(T _val, size_t _n){
        Fill(_val, _n, 0);
    }

    void Fill(T _val){
        Fill(_val, n, 0);
    }

    void ReadIn(thrust::host_vector<T> _src, size_t _n, size_t _offset1,
                size_t _offset2){
        thrust::copy(_src.begin()+_offset1, _src.begin()+_offset1+_n,
                     h.begin()+_offset2);
        CopyToDevice(_n, _offset2);
    }

    void ReadIn(thrust::host_vector<T> _src){
        ReadIn(_src, _src.size(), 0, 0);
    }

};

template<typename T>
struct SimList3D{
    size_t n;

    SimList1D<T> x; 
    SimList1D<T> y; 
    SimList1D<T> z;

    R3Nptrs devPtrs; 
    R3Nptrs hostPtrs; 
    
    SimList3D(size_t _n): n(_n), x(_n), y(_n), z(_n){
        devPtrs.x = x.devPtr;
        devPtrs.y = y.devPtr;
        devPtrs.z = z.devPtr;
        
        hostPtrs.x = x.hostPtr;
        hostPtrs.y = y.hostPtr;
        hostPtrs.z = z.hostPtr;
    }
    
    ~SimList3D(){
        devPtrs.x = NULL;
        devPtrs.y = NULL;
        devPtrs.z = NULL;
        
        hostPtrs.x = NULL;
        hostPtrs.y = NULL;
        hostPtrs.z = NULL;
    }

    // copy constructor
    SimList3D(const SimList3D& other): n(other.n), x(other.x), y(other.y),
                                       z(other.z){
    }

    // copy assignment
    SimList3D& operator=(const SimList3D& other){
        n = other.n;
        x = other.x;
        y = other.y;
        z = other.z; 
    }

    void CopyToDevice(size_t _n, size_t offset){
        x.CopyToDevice(_n, offset);
        y.CopyToDevice(_n, offset);
        z.CopyToDevice(_n, offset);
    }

    void CopyToDevice(){
        x.CopyToDevice(n, 0);
        y.CopyToDevice(n, 0);
        z.CopyToDevice(n, 0);
    }

    void CopyToHost(size_t _n, size_t offset){
        x.CopyToHost(_n, offset);
        y.CopyToHost(_n, offset);
        z.CopyToHost(_n, offset);
    }

    void CopyToHost(){
        x.CopyToHost(n, 0);
        y.CopyToHost(n, 0);
        z.CopyToHost(n, 0);
    }

};
#endif // SimList_CUH
