#include <CL/cl.h>  // Include the standard C OpenCL header
#include <iostream>
#include <vector>

int run() {
    // Step 1: Get all available platforms
    cl_uint num_platforms;
    cl_int result = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (result != CL_SUCCESS || num_platforms == 0) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return 1;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    // Step 2: Iterate through each platform
    for (const auto& platform : platforms) {
        char platform_name[128];
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
        std::cout << "Platform: " << platform_name << std::endl;

        // Step 3: Get devices for this platform
        cl_uint num_devices;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
        if (num_devices == 0) {
            std::cout << "  No devices found for this platform." << std::endl;
            continue;
        }

        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);

        // Step 4: Print device information
        for (const auto& device : devices) {
            char device_name[128];
            clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
            cl_uint compute_units;
            clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
            cl_ulong global_mem_size;
            clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, nullptr);

            std::cout << "  Device: " << device_name << std::endl;
            std::cout << "    Max Compute Units: " << compute_units << std::endl;
            std::cout << "    Global Memory Size: " << (global_mem_size / (1024 * 1024)) << " MB" << std::endl;
        }
    }

    return 0;
}
