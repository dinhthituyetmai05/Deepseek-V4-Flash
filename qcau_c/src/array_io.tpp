#include "../include/Core.h"
#include <iomanip>
#include <fstream>


template<typename T>
void read_arr(const char* file_path, T* arr, int size) {
    std::ifstream file(file_path);
    float val;
    int i = 0;
    while (file >> val && i < size) {
        arr[i++] = static_cast<T>(val);
    }
}
template<typename T>
void write_arr(const char* file_path, T* arr, int size, unsigned int precision) {
    std::ofstream file(file_path);
    float val;
    for (int i = 0; i < size; i++) {
        val = (float)arr[i];
        file << std::fixed << std::setprecision(precision) << val << std::endl;
    }
    file.close();
}
