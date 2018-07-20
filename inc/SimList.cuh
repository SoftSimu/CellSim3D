#ifndef SimList_CUH
#define SIMLIST_CUH
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
// #include <thrust/copy.h>
#include <cstdlib>
#include <string>
#include <exception>
#include "Types.cuh"
struct base_n{
    static size_t used_host_mem;
};

template<typename T>
struct SimList1D: base_n{
    size_t n;

    T* d;
    T* h; 

    T* devPtr;
    T* hostPtr;

    // This is proper constructor delegation in C++11
    SimList1D(size_t _n): SimList1D(_n, T(0)){}
    
    SimList1D(size_t _n, T _val)
    try : n(_n){
        base_n::used_host_mem += n*sizeof(T);

        if (cudaMalloc((void **)&devPtr, sizeof(T)*n) != cudaSuccess){
            std::cout << "memory allocation error" << std::endl;
        }
        
        d = devPtr;

        hostPtr = new T[n]; 
        h = hostPtr;

        for (size_t i = 0; i < n; ++i){
            h[i] = _val; 
        }

        CopyToDevice();

    } catch (const std::exception& e){
        std::cout << "Bad Memory Exception while trying to allocate "
                  << sizeof(T)*n/(1024*1024) << " GB." << std::endl;
        std::cout << e.what() << std::endl; 
        throw e; 
    }

    ~SimList1D(){
        base_n::used_host_mem -= n*sizeof(T);
        cudaFree(devPtr);
        
        delete[] hostPtr;
        h = NULL; hostPtr=NULL;
    }

    // copy constructor
    SimList1D(const SimList1D& other): n(other.n){
        // h = other.h;
        // hostPtr = other.hostPtr;

        // d = other.d;
        // devPtr = other.devPtr;
        base_n::used_host_mem += n*sizeof(T);

        cudaMalloc((void **)&devPtr, sizeof(T)*n);
        d = devPtr;
        cudaMemcpy(devPtr, other.devPtr, n*sizeof(T), cudaMemcpyDeviceToDevice);

        hostPtr = new T[n]; 
        h = hostPtr;

        std::memcpy(hostPtr, other.hostPtr, n*sizeof(T));

    }
    
    // copy assignment
    SimList1D& operator=(const SimList1D& other){        
        if (this != &other){
            n = other.n;
            // h = other.h;
            // hostPtr = other.hostPtr;

            // d = other.d;
            // devPtr = other.devPtr;
            base_n::used_host_mem += n*sizeof(T);

            cudaMalloc(&devPtr, sizeof(T)*n);
            cudaMemcpy(devPtr, other.devPtr, n*sizeof(T), cudaMemcpyDeviceToDevice); 
            d = devPtr;

            delete[] hostPtr; 
            hostPtr = NULL; h = NULL;
            
            hostPtr = new T[n];
            h = hostPtr;

            std::memcpy(hostPtr, other.hostPtr, n*sizeof(T));
        }
        return *this;
    }    

    void CopyToDevice(const size_t& _n, const size_t& offset){
        //thrust::copy(h.begin()+offset, h.begin()+offset+_n, d.begin()+offset);
        cudaMemcpy((void *)(d+offset), (void *)(h+offset), _n*sizeof(T), cudaMemcpyHostToDevice);
    }

    void CopyToDevice(){
        CopyToDevice(n, 0);
    }

    void CopyToHost(const size_t& _n, const size_t& offset){
        //thrust::copy(d.begin()+offset, d.begin()+offset+_n, h.begin()+offset);
        cudaMemcpy((void *)(h+offset), (void *)(d+offset), _n*sizeof(T), cudaMemcpyDeviceToHost);
    }

    void CopyToHost(){
        CopyToHost(n, 0);
    }

    void Fill(const T& _val, const size_t& _n, const size_t& offset){
        //thrust::fill(h.begin()+offset, h.begin()+offset+_n, _val);
        for (size_t i = 0; i < _n; ++i){
            h[i+offset] = _val;
        }
        
        CopyToDevice(_n, offset);
    }

    void Fill(T _val, size_t _n){
        Fill(_val, _n, 0);
    }

    void Fill(T _val){
        Fill(_val, n, 0);
    }

    void ReadIn(T* _src, const size_t& _n, const size_t& _offsetSrc,
                const size_t& _offsetDst){

        for (size_t i = 0; i < _n; ++i){
            h[i+_offsetDst] = _src[i+_offsetSrc];
        }
        
        CopyToDevice(_n, _offsetDst);
    }

    void ReadIn(T* _src, size_t _n){
        ReadIn(_src, _n, 0, 0);
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
