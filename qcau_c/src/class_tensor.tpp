#include "../include/Core.h"



template <typename T>
Tensor<T>::Tensor() : data(NULL), own_memory(false) {}
template <typename T>
Tensor<T>::Tensor(T* data, const Shape &shape, bool own_memory) 
            : data(data), shape(shape), own_memory(own_memory) {}
template <typename T>
Tensor<T>::Tensor(const Shape &shape) : shape(shape), own_memory(true) {
    data = new T[shape.S* shape.H* shape.C]{};
}
template <typename T>
Tensor<T>::~Tensor() {
    if (own_memory && data)
        delete[] data;
}
template <typename T>
inline int Tensor<T>::index(int s, int h, int c) {
    return (s* shape.H + h)* shape.C + c;
}
template <typename T>
void Tensor<T>::print() {
    int size = shape.S* shape.H* shape.C;
    for (int i = 0; i < size; i++) std::cout << static_cast<float>(data[i]) << " ";
    std::cout << "\n";
}
template <typename T>
inline void Tensor<T>::reshape(int s, int h, int c) {
    if (s * h * c != shape.S * shape.H * shape.C) 
        assert(0 && "Error: Inappropriate shape for reshape !!");
    shape.S = s, shape.H = h, shape.C = c;
}
template <typename T>
inline T Tensor<T>::get(int s, int h, int c) {
    return data[index(s, h, c)];
}
template <typename T>
inline T &Tensor<T>::at(int s, int h, int c) {
    return data[index(s, h, c)];
}
template <typename T>
inline T* Tensor<T>::raw_at(int s, int h, int c) {
    return &data[index(s, h, c)];
}
