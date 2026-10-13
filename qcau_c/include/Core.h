#ifndef TENSOR_MEM_H
#define TENSOR_MEM_H

#include <iostream>
#include <cassert>
#include <cmath>
#include <cinttypes>


struct Shape {
    int S; // Seq
    int H; // Head
    int C; // Channels
    /**
     * @brief New Shape object with full 0-sizes
     * 
     */
    Shape() : S(0), H(0), C(0) {}
    /**
     * @brief New Shape object
     * 
     * @param s Seq
     * @param h Head
     * @param c Channels
     */
    Shape(int s, int h, int c) : S(s), H(h), C(c) {}

    bool operator==(const Shape &other) const {
        return S == other.S && H == other.H && C == other.C;
    }
};

template <typename T>
class Tensor {
private:
    T* data;
    bool own_memory;
    inline int index(int s, int h, int c);

public:
    Shape shape;

    /**
     * @brief New Tensor with no mem
     */
    Tensor();
    /**
     * @brief New Tensor
     * 
     * @param data 
     * @param shape {S, H, C}
     * @param is_own_memory 
     */
    Tensor(T* data, const Shape &shape, bool own_memory);
    /**
     * @brief New Tensor with own full 0s mem
     * 
     * @param shape {S, H, C}
     */
    Tensor(const Shape &shape);
    ~Tensor();

    /**
     * @brief print data of tensor
     * 
     */
    void print();

    /**
     * @brief get tensor value at position
     * 
     * @param s 
     * @param h 
     * @param c 
     */
    inline void reshape(int s, int h, int c);

    /**
     * @brief get tensor value at position
     * 
     * @param s 
     * @param h 
     * @param c 
     * @return (T) value 
     */
    inline T get(int s, int h, int c);

    /**
     * @brief get tensor reference at position
     * 
     * @param s
     * @param h  
     * @param c 
     * @return (T) reference 
     */
    inline T &at(int s, int h, int c);

    /**
     * @brief raw data
     * 
     * @return T* 
     */
    inline T* raw() { return data; }
    /**
     * @brief raw data
     * 
     * @return const T* 
     */
    inline const T* raw() const { return data; }
    inline T* raw_at(int s, int h, int c);
};

#include "../src/class_tensor.tpp"
#include "../src/tensor_io.tpp"
#include "../src/array_io.tpp"

#endif
