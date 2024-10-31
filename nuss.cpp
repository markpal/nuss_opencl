#include <CL/cl.h>
#include <iostream>
#include <cstdlib>   // for random numbers
#include <omp.h>
#include <vector>
#include <cstring>   // for strcpy
#include <string>
#include <ctime>     // for time()
#include <climits>
#include "intel_ocl.h"

#define BLOCK_SIZE 16
int N = 1024;

using namespace std;

// -------------------------------------------------- pairing function
int paired(char a1, char a2) {
    if(a1 == 'A' && a2 == 'U') return 1;
    if(a1 == 'U' && a2 == 'A') return 1;
    if(a1 == 'G' && a2 == 'C') return 1;
    if(a1 == 'C' && a2 == 'G') return 1;
    return 0;
}

// -------------------------------------------------- main function

int main() {
    // Initialize sequence
    string seq = "GUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUAC";
    //seq = "CUGGUUUAUGUCACCCAGCAGCAGACCCUCCUUUACCGAAAGAUGAUGCUCGUAUUAUUGUACG";
    //N += BLOCK_SIZE - N % BLOCK_SIZE;

   //  N = seq.length();

     //run();

    char* seqq = new char[N];
    char znaki[] = {'C', 'G', 'U', 'A'};
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < N; i++) {
        seqq[i] = znaki[rand() % 4];
    }
    //std::strcpy(seqq, seq.c_str());

    cout << seqq;

    int* flatArray_S = new int[N * N];
    int* flatArray_S_CPU = new int[N * N];

    // Initialization
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            flatArray_S[i * N + j] = INT_MIN;
            flatArray_S_CPU[i * N + j] = INT_MIN;
        }
    }
    for (int i = 0; i < N; i++) {
        flatArray_S[i * N + i] = 0;
        flatArray_S_CPU[i * N + i] = 0;
        if (i + 1 < N) {
            flatArray_S[i * N + i + 1] = 0;
            flatArray_S[i * N + 1 + i] = 0;
            flatArray_S_CPU[i * N + i + 1] = 0;
            flatArray_S_CPU[i * N + 1 + i] = 0;
        }
    }

    FILE* file = fopen("kernel.cl", "r");
    if (!file) {
        fprintf(stderr, "Error opening file.\n");
        return 1;
    }

    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    rewind(file);

    char* kernelSource = (char*)malloc(fileSize + 1);
    if (!kernelSource) {
        fclose(file);
        fprintf(stderr, "Memory allocation error.\n");
        return 1;
    }

    const char* sources[] = { kernelSource };

    fread(kernelSource, 1, fileSize, file);
    fclose(file);
    kernelSource[fileSize] = '\0';

    cl_int err;
    cl_platform_id cpPlatform[10];
    cl_uint platf_num;
    cl_device_id device;
    err = clGetPlatformIDs(10, cpPlatform, &platf_num);

    for(int i = 0; i < platf_num; i++)
        if(0 == (err = clGetDeviceIDs(cpPlatform[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL))) {
            printf("Platform #%d\n", i);
            break;
        }
    printf("GetID=%d\n", err);

    cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    printf("CreateCommandQueue=%d\n", err);

    cl_program program = clCreateProgramWithSource(context, 1, sources, NULL, &err);

    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_int build_status;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_int), &build_status, NULL);

    if (build_status != CL_SUCCESS) {
        size_t log_size;
        // Print compilation errors
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *program_log = (char *)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        program_log[log_size] = '\0';
        printf("Compilation Log:\n%s\n", program_log);
        free(program_log);

        // Handle the compilation error as needed
        return 1; // or some other error code
    }

    cl_kernel kernel = clCreateKernel(program, "myKernel", &err);



    cl_mem d_sequence = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(int), NULL, &err);
    clEnqueueWriteBuffer(queue, d_sequence, CL_TRUE, 0,  N * sizeof(char), seqq, 0, NULL, NULL);

    cl_mem d_flat_S = clCreateBuffer(context, CL_MEM_READ_WRITE, N * N * sizeof(int), NULL, &err);
    clEnqueueWriteBuffer(queue, d_flat_S, CL_TRUE, 0, N * N * sizeof(int), flatArray_S, 0, NULL, NULL);


    // Kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_flat_S);
   err |= clSetKernelArg(kernel, 1, sizeof(int), &N);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_sequence);


    // Define work sizes
    size_t localSize[2] = { (size_t)BLOCK_SIZE, (size_t)BLOCK_SIZE };
    size_t globalSize[2] = { (size_t)BLOCK_SIZE, (size_t)BLOCK_SIZE };
    int bb = BLOCK_SIZE;
    // Start timing
    double start_time = omp_get_wtime();

    cl_event kernelEvent;

    // Launch kernel across multiple loop iterations
    for (int c0 = 0; c0 <= (N - 1) / BLOCK_SIZE; c0 += 1) {
        err = clSetKernelArg(kernel, 3, sizeof(int), &c0);
        if (err != CL_SUCCESS)
            std::cerr << "Failed to send arg: " << err << std::endl;

        int numBlocks = min((N - 1) / bb, (N + c0 - 2 )/ bb) - c0 + 1;
        globalSize[0] = (size_t)numBlocks*BLOCK_SIZE;
        globalSize[1] = BLOCK_SIZE;

        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            std::cerr << "Error enqueuing kernel: " << err << std::endl;
            // Cleanup
            return -1;
        }

        cl_int kernelStatus;
        clGetEventInfo(kernelEvent, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &kernelStatus, nullptr);
        if (kernelStatus < 0) {
            std::cerr << "Kernel execution failed with status: " << kernelStatus << std::endl;
        } else {
           // std::cout << "Kernel executed successfully!" << std::endl;
        }


        clFinish(queue);
    }



    // Check the event status for errors


    // Copy result back to host
    err = clEnqueueReadBuffer(queue, d_flat_S, CL_TRUE, 0, N * N * sizeof(int), flatArray_S, 0, nullptr, nullptr);

    // Stop timing
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    printf("Time taken: %f seconds\n", elapsed_time);

    printf("gpu ended\n");

    // CPU verification
    for (int i = N - 1; i >= 0; i--) {
        for (int j = i + 1; j < N; j++) {
            for (int k = 0; k < j - i; k++) {
                flatArray_S_CPU[i * N + j] = max(flatArray_S_CPU[i * N + k + i] + flatArray_S_CPU[(k + i + 1) * N + j], flatArray_S_CPU[i * N + j]);
            }
            flatArray_S_CPU[i * N + j] = max(flatArray_S_CPU[i * N + j], flatArray_S_CPU[(i + 1) * N + j - 1] + paired(seqq[i], seqq[j]));
        }
    }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (flatArray_S[i * N + j] != flatArray_S_CPU[i * N + j]) {
                cout << i << " " << j << ":" << flatArray_S[i * N + j] << " " << flatArray_S_CPU[i * N + j] << endl;
                cout << "error" << endl;
               // exit(1);
            }

    cout << endl << endl;
    //
   if(1==0)
        for(int i=0; i<N; i++){
            for(int j=0; j<N; j++){
                if(flatArray_S[i * N + j] < 0)
                    cout << "";
                else
                    cout << flatArray_S[i * N + j];
                cout << "\t";
            }
            cout << "\n";
        }
    cout << endl;

    cout << endl << endl;
    //
    if(1==0)
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            if(flatArray_S_CPU[i * N + j] < 0)
                cout << "";
            else
                cout << flatArray_S_CPU[i * N + j];
            cout << "\t";
        }
        cout << "\n";
    }
    cout << endl;

    // Cleanup
    delete[] seqq;
    delete[] flatArray_S;
    delete[] flatArray_S_CPU;
    clReleaseMemObject(d_sequence);
    clReleaseMemObject(d_flat_S);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
