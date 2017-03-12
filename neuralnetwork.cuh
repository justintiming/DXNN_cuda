#ifndef neuralnetwork
#define neuralnetwork

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

class nn
{
public:

    int networkID;

    // functions
    nn();
    void initialize(int in, int out, float wrange, float brange, int activationFuncCount, int nID);
    void sort(cudaStream_t &stream);
    void pretun(cudaStream_t &stream);
    void propagate(float* inputArray, cudaStream_t &stream, int tun, int offset);
    void mutateTopology();
    void setWeightRange(float weightRange);
    void gen_rand_num(cudaStream_t &stream, int size);

    // initializing data
    int inputDim, outputDim;

    // tuning phase
    int numWeightsfromNGN;
    int ngnptr;
    // mutate phase TODO

    // parameters
    float weightRange;
    float biasRange;

    // neurons
    thrust::host_vector<int, thrust::cuda::experimental::pinned_allocator<int> > neuronsID;
    thrust::host_vector<float, thrust::cuda::experimental::pinned_allocator<float> > values;
    thrust::host_vector<float, thrust::cuda::experimental::pinned_allocator<float> > bias;
    thrust::host_vector<int, thrust::cuda::experimental::pinned_allocator<int> > activationFunc;
    thrust::host_vector<int, thrust::cuda::experimental::pinned_allocator<int> > layer;
    thrust::host_vector<int, thrust::cuda::experimental::pinned_allocator<int> > perNeuronLayer;
    thrust::host_vector<int, thrust::cuda::experimental::pinned_allocator<int> > generation;
    thrust::host_vector<int, thrust::cuda::experimental::pinned_allocator<int> > perNeuronWeights;

    // network in COO format
    thrust::host_vector<float, thrust::cuda::experimental::pinned_allocator<float> > weights;
    thrust::host_vector<float, thrust::cuda::experimental::pinned_allocator<float> > temp_weights;
    thrust::host_vector<int, thrust::cuda::experimental::pinned_allocator<int> > from;
    thrust::host_vector<int, thrust::cuda::experimental::pinned_allocator<int> > to;

    // randoms
    thrust::host_vector<float, thrust::cuda::experimental::pinned_allocator<float> > random1;
    thrust::host_vector<float, thrust::cuda::experimental::pinned_allocator<float> > random2;

    //statistics
    int numNeurons;
    int numWeights;
    int numLayers;
    int numActivationFunc;
    int currentGeneration;

    int num_iteration;
};
#endif
