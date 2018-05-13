#include "core/utils.hpp"


int main(int argc, char **argv) {
    // Test Random
    {
        std::vector<float> vec (100, 0);
        fill_random(vec);
        std::sort(vec.begin(), vec.end());
        const auto mean = std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
        std::cout << "Test Random vector mean is: " << mean << ", min=" << vec.front() << ", max=" << vec.back() << std::endl;
    }
    // Test To String
    {
        std::cout << "Test to String: int=" << S((int)1) << ", float=" << S((float)2.343) << 
                                      ", bool=" << S(true) << ", bool=" << S(false) << std::endl;
    }
    // Test Format
    {
        std::cout << fmt("TestFormat: Hello %s", "world!") << std::endl;
        std::cout << fmt("TestFormat: Hello robot number %d", 1) << std::endl;
        std::cout << fmt("TestFormat: Floating point number is %.3f", 3.462464) << std::endl;
        std::cout << fmt("TestFormat: boolean value is %s", "false") << std::endl;
        const std::string s = "str instance";
        std::cout << fmt("TestFormat: Hello %s", s.c_str()) << std::endl;
    }
    // Test Sharded Vector
    {
        std::vector<int> vec(1000, 0);
        std::cout << "Test Sharded Vector: " << sharded_vector<int>(vec,  1,  0) << std::endl;
        std::cout << "Test Sharded Vector: " << sharded_vector<int>(vec,  2,  0) << std::endl;
        std::cout << "Test Sharded Vector: " << sharded_vector<int>(vec,  2,  1) << std::endl;
        std::cout << "Test Sharded Vector: " << sharded_vector<int>(vec, 33, 32) << std::endl;
    }
    // Test Running Average
    {
        running_average ra;
        for (int i=1; i<=10; ++i)
            ra.update(i);
        std::cout << "Test Running Average: " << ra << std::endl;
    }
}