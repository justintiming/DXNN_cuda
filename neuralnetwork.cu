#include "neuralnetwork.cuh"
#include "activationFunc.cuh"
#include <iostream>
#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/system/cuda/execution_policy.h>
#include "helper_cuda.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/count.h>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cstdlib>
#include <ctime>

// Utilities and system includes
#include <cub/cub.cuh>
// before mem optimization
nn::nn()
{
	inputDim = 0;
	outputDim = 0;
	weightRange = 0;
	biasRange = 0;
	networkID = 0;
	numNeurons = 0;
	numWeights = 0;
	numLayers = 0;
	currentGeneration = 0;
	numActivationFunc = 0;
	num_iteration = 1;
	numWeightsfromNGN = 0;
	ngnptr = 0;
}

void nn::initialize(int in, int out,
                    float wrange, float brange,
                    int activationFuncCount, int nID)
{
    inputDim = in;
    outputDim = out;
    weightRange = wrange;
    biasRange = brange;
    networkID = nID;
    numNeurons = in + 2*out;
    numWeights = in*out + out;
    numLayers = 1;
    numActivationFunc = activationFuncCount;
    currentGeneration = 0;

    // neuronsID, layer, compressed layer, per neuron layer
    for (int i = 0; i < numNeurons; i++)
    {
    	neuronsID.push_back(i);
    	if (i < in)
    	{
    		perNeuronLayer.push_back(0);
    	}
    	else if (i < in + out)
    	{
    		perNeuronLayer.push_back(1);
    	}
    	else
    	{
    		perNeuronLayer.push_back(2);
    	}
    }

    for (int i = 0; i < numWeights; i++)
    {
        if (i < in*out)
        {
            layer.push_back(1);
        }
        else
        {
            layer.push_back(2);
        }
    }

    float zerof = 0.0;
    int zeroi = 0;
    size_t size = numNeurons;
    // values, bias and generation
    values.assign(size, zerof);
    bias.assign(size, zerof);
    generation.assign(size, zeroi);
    // activationFunc
    activationFunc.assign(size, zeroi);
    perNeuronWeights.assign(size, zeroi);
    std::srand(time(NULL));
    for (int i = in; i < in + out; i++)
    {
        activationFunc[i] = rand()%numActivationFunc;
    }

    // COO
    // weights
    for (int i = 0; i < in*out; i++)
    {
    	float w = (std::rand()*weightRange)/RAND_MAX;

        if (std::rand()%2 == 0)
        {
            weights.push_back(w);
        }
        else
        {
        	weights.push_back(-w);
        }
    }
    for (int i = 0; i < out; i++)
    {
        weights.push_back(1);
    }

    // from i.e. colIdx
    for (int i = 0; i < out; i++)
    {
        for (int j = 0; j < in; j++)
        {
            from.push_back(j);
        }
    }
    for (int i = 0; i < out; i++)
    {
        from.push_back(i + in); // hidden layer
    }

    // to i.e. rowIdx
    for (int i = 0; i < out; i++)
    {
        for (int j = 0; j < in; j++)
        {
            to.push_back(i + in);
        }
    }
    for (int i = 0; i < out; i++)
    {
        to.push_back(i + in + out);
    }
}

// propagate network

struct replace
{
	__host__ __device__
	float operator()(float& x, float& y)
	{
		return y;
	}
};
void nn::sort(cudaStream_t &stream)
{
	// layer
    thrust::device_vector<int> d_layer(numWeights);
    thrust::device_vector<float> d_weights(numWeights);
    thrust::device_vector<int> d_from(numWeights);
    thrust::device_vector<int> d_to(numWeights);

    thrust::device_vector<int> d_layer_o(numWeights);
    thrust::device_vector<float> d_weights_o(numWeights);
    thrust::device_vector<int> d_from_o(numWeights);
    thrust::device_vector<int> d_to_o(numWeights);

    thrust::device_vector<int> d_indice(numWeights);
    thrust::sequence(thrust::cuda::par.on(stream), d_indice.begin(), d_indice.end());
    thrust::device_vector<int> d_indice_o(numWeights);

	// for activationFunc
    thrust::device_vector<int> d_neuronsID(numNeurons);
    thrust::device_vector<int> d_generation(numNeurons);
    thrust::device_vector<float> d_values(numNeurons);
    thrust::device_vector<float> d_bias(numNeurons);
    thrust::device_vector<int> d_activationFunc(numNeurons);
    thrust::device_vector<int> d_perNeuronLayer(numNeurons);
    thrust::device_vector<int> d_index(numNeurons);
    thrust::sequence(thrust::cuda::par.on(stream), d_index.begin(), d_index.end());

    thrust::device_vector<int> d_neuronsID_o(numNeurons);
    thrust::device_vector<int> d_generation_o(numNeurons);
    thrust::device_vector<float> d_values_o(numNeurons);
    thrust::device_vector<float> d_bias_o(numNeurons);
    thrust::device_vector<int> d_activationFunc_o(numNeurons);
    thrust::device_vector<int> d_perNeuronLayer_o(numNeurons);
    thrust::device_vector<int> d_index_o(numNeurons);

    // mem transfer
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_layer.data()),            thrust::raw_pointer_cast(layer.data()),             numWeights*sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_weights.data()),          thrust::raw_pointer_cast(weights.data()),           numWeights*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_from.data()),             thrust::raw_pointer_cast(from.data()),              numWeights*sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_to.data()),               thrust::raw_pointer_cast(to.data()),                numWeights*sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_neuronsID.data()),        	thrust::raw_pointer_cast(neuronsID.data()),         numNeurons*sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_generation.data()),        	thrust::raw_pointer_cast(generation.data()),         numNeurons*sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_values.data()),           	thrust::raw_pointer_cast(values.data()),            numNeurons*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_bias.data()),             	thrust::raw_pointer_cast(bias.data()),              numNeurons*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_activationFunc.data()),   	thrust::raw_pointer_cast(activationFunc.data()),    numNeurons*sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_perNeuronLayer.data()),   	thrust::raw_pointer_cast(perNeuronLayer.data()),    numNeurons*sizeof(int), cudaMemcpyHostToDevice, stream);

    int* d_indice_in = thrust::raw_pointer_cast(d_indice.data());
    int* d_indice_out = thrust::raw_pointer_cast(d_indice_o.data());
    int* d_layer_in = thrust::raw_pointer_cast(d_layer.data());
    int* d_layer_out = thrust::raw_pointer_cast(d_layer_o.data());

    size_t temp_storage_bytes = 0;
    void* temp_storage = NULL;
    cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_bytes, d_layer_in, d_layer_out, d_indice_in, d_indice_out, numWeights, 0, sizeof(int)*8, stream);
    cudaMalloc(&temp_storage, temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_bytes, d_layer_in, d_layer_out, d_indice_in, d_indice_out, numWeights, 0, sizeof(int)*8, stream);

    cudaFree(temp_storage);

    thrust::gather(thrust::cuda::par.on(stream), d_indice_o.begin(), d_indice_o.end(), thrust::make_zip_iterator(thrust::make_tuple(d_weights.begin(), d_from.begin(), d_to.begin())), thrust::make_zip_iterator(thrust::make_tuple(d_weights_o.begin(), d_from_o.begin(), d_to_o.begin())));

    int* d_activationFunc_in = thrust::raw_pointer_cast(d_activationFunc.data());
    int* d_perNeuronLayer_in = thrust::raw_pointer_cast(d_perNeuronLayer.data());
    int* d_index_in = thrust::raw_pointer_cast(d_index.data());
    int* d_activationFunc_out = thrust::raw_pointer_cast(d_activationFunc_o.data());
    int* d_perNeuronLayer_out = thrust::raw_pointer_cast(d_perNeuronLayer_o.data());
    int* d_index_out = thrust::raw_pointer_cast(d_index_o.data());

    d_perNeuronLayer_o.assign(d_perNeuronLayer.begin(), d_perNeuronLayer.end());

    temp_storage_bytes = 0;
    temp_storage = NULL;
    cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_bytes, d_activationFunc_in, d_activationFunc_out, d_index_in, d_index_out, numNeurons, 0, sizeof(int)*8, stream);
    cudaMalloc(&temp_storage, temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_bytes, d_activationFunc_in, d_activationFunc_out, d_index_in, d_index_out, numNeurons, 0, sizeof(int)*8, stream);
    d_index.assign(d_index_o.begin(), d_index_o.end());
    thrust::gather(thrust::cuda::par.on(stream), d_index.begin(), d_index.end(), d_perNeuronLayer_o.begin(), d_perNeuronLayer.begin());
    cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_bytes, d_perNeuronLayer_in, d_perNeuronLayer_out, d_index_in, d_index_out, numNeurons, 0, sizeof(int)*8, stream);

    cudaFree(temp_storage);

    thrust::gather(thrust::cuda::par.on(stream), d_index_o.begin(), d_index_o.end(), thrust::make_zip_iterator(thrust::make_tuple(d_neuronsID.begin(), d_generation.begin(), d_values.begin(), d_bias.begin(), d_activationFunc.begin())), thrust::make_zip_iterator(thrust::make_tuple(d_neuronsID_o.begin(), d_generation_o.begin(), d_values_o.begin(), d_bias_o.begin(), d_activationFunc_o.begin())));

    cudaMemcpyAsync(thrust::raw_pointer_cast(neuronsID.data()),        thrust::raw_pointer_cast(d_neuronsID_o.data()),         numNeurons*sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(generation.data()),        thrust::raw_pointer_cast(d_generation_o.data()),         numNeurons*sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(values.data()),           thrust::raw_pointer_cast(d_values_o.data()),            numNeurons*sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(bias.data()),             thrust::raw_pointer_cast(d_bias_o.data()),              numNeurons*sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(activationFunc.data()),   thrust::raw_pointer_cast(d_activationFunc_o.data()),    numNeurons*sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(perNeuronLayer.data()),   thrust::raw_pointer_cast(d_perNeuronLayer_o.data()),    numNeurons*sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(layer.data()),        thrust::raw_pointer_cast(d_layer_o.data()),      numWeights*sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(weights.data()),      thrust::raw_pointer_cast(d_weights_o.data()),    numWeights*sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(from.data()),         thrust::raw_pointer_cast(d_from_o.data()),       numWeights*sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(to.data()),           thrust::raw_pointer_cast(d_to_o.data()),         numWeights*sizeof(int), cudaMemcpyDeviceToHost, stream);
}

struct pre_reduction
{
	int currentgen;

	pre_reduction(int a)
	{
		currentgen = a;
	}
	__host__ __device__
	int operator()(int& x)
	{
		if (x < currentgen - 2)
		{
			return 0;
		}
		else
		{
			return 1;
		}
	}
};

void nn::pretun(cudaStream_t &stream)
{
	thrust::device_vector<int> d_generation(numNeurons);
	thrust::device_vector<int> d_temp(numNeurons);
	thrust::device_vector<int> d_from(numWeights);
	thrust::device_vector<int> d_histogram(numNeurons);

    cudaMemcpyAsync(thrust::raw_pointer_cast(d_generation.data()), thrust::raw_pointer_cast(generation.data()), numNeurons*sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_from.data()),       thrust::raw_pointer_cast(from.data()),       numWeights*sizeof(int), cudaMemcpyHostToDevice, stream);

    thrust::transform(thrust::cuda::par.on(stream), d_generation.begin(), d_generation.end(), d_temp.begin(), pre_reduction(currentGeneration));

    int currentgen = thrust::count(thrust::cuda::par.on(stream), d_temp.begin(), d_temp.end(), 1);
    int recentgen = static_cast<int>(std::sqrt(std::abs(numNeurons - currentgen)));
    ngnptr = numNeurons - currentgen - recentgen;

    int* samples = thrust::raw_pointer_cast(d_from.data());
    int* histogram = thrust::raw_pointer_cast(d_histogram.data());
    void*    d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, samples, histogram, numNeurons+1, 0, numNeurons, numWeights);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, samples, histogram, numNeurons+1, 0, numNeurons, numWeights);

    cudaFree(d_temp_storage);

    numWeightsfromNGN = thrust::reduce(thrust::cuda::par.on(stream), d_histogram.begin() + ngnptr, d_histogram.end());

    cudaMemcpyAsync(thrust::raw_pointer_cast(perNeuronWeights.data()), thrust::raw_pointer_cast(d_histogram.data()), numNeurons*sizeof(int), cudaMemcpyDeviceToHost, stream);
}

typedef thrust::tuple<int, float> tuple;

struct tun1
{
	int ngnptr;
	int numNeurons;
	tun1(int _ngnptr, int _numNeurons)
	{
		ngnptr = _ngnptr;
		numNeurons = _numNeurons;
	}
	__host__ __device__
	int operator()(int& NGN, float& random)
	{
		if(NGN >= ngnptr && random <= 1.0f/sqrtf(static_cast<float>(numNeurons - ngnptr)))
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}
};

struct tun2
{
	int* NGN;
	int* histogram;
	float wrange;
	tun2(int* _NGN, int* _histogram, float _wrange)
	{
		NGN = _NGN;
		histogram = _histogram;
		wrange = _wrange;
	}
	__host__ __device__
	float operator()(float& weights, const tuple& random)
	{
		int i = thrust::get<0>(random);
		float j = thrust::get<1>(random);
		if (NGN[i] == 1 && j <= 1.0f/sqrtf(static_cast<float>(histogram[i])))
		{
			return wrange*(2*j - 1);
		}
		else
		{
			return weights;
		}

	}
};

void nn::propagate(float* inputArray, cudaStream_t &stream, int tun, int offset)
{
	values.assign(inputArray, inputArray + inputDim);

	cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);
    cusparseSetStream(cusparseHandle, stream);

    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);

    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    thrust::device_vector<int> d_layer_offset(numLayers + 2);
    thrust::device_vector<int> d_perNeuronLayer_offset(numLayers + 2);
    thrust::device_vector<float> d_weights(numWeights);
	thrust::device_vector<int> d_from(numWeights);
	thrust::device_vector<int> d_to(numWeights);
    thrust::device_vector<float> d_values(numNeurons);
    thrust::device_vector<int> d_neuronsID(numNeurons);
    thrust::device_vector<int> d_layer(numWeights);
    thrust::device_vector<int> d_perNeuronLayer(numNeurons);
    thrust::device_vector<int> d_activationFunc(numNeurons);
    thrust::device_vector<float> d_bias(numNeurons);

    cudaMemcpyAsync(thrust::raw_pointer_cast(d_weights.data()),     thrust::raw_pointer_cast(weights.data()),       numWeights*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_from.data()),        thrust::raw_pointer_cast(from.data()),          numWeights*sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_values.data()),      thrust::raw_pointer_cast(values.data()),        numNeurons*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_neuronsID.data()),   thrust::raw_pointer_cast(neuronsID.data()),     numNeurons*sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_layer.data()),       thrust::raw_pointer_cast(layer.data()),         numWeights*sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_perNeuronLayer.data()),    thrust::raw_pointer_cast(perNeuronLayer.data()),     numNeurons*sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_activationFunc.data()),    thrust::raw_pointer_cast(activationFunc.data()),     numNeurons*sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_bias.data()),        thrust::raw_pointer_cast(bias.data()),          numNeurons*sizeof(float), cudaMemcpyHostToDevice, stream);

    if (tun == 1)
    {
    	temp_weights.assign(numWeights, 0);
        thrust::device_vector<int> d_perNeuronWeights(numNeurons);
        thrust::device_vector<float> d_random1(numNeurons);
        thrust::device_vector<float> d_random2(numWeights);
        thrust::device_vector<int> d_NGN(numNeurons);
        thrust::sequence(thrust::cuda::par.on(stream), d_NGN.begin(), d_NGN.end());

        cudaMemcpyAsync(thrust::raw_pointer_cast(d_random1.data()),     thrust::raw_pointer_cast(random1.data()) + offset*numNeurons, numNeurons*sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(thrust::raw_pointer_cast(d_random2.data()),     thrust::raw_pointer_cast(random2.data()) + offset*numWeights, numWeights*sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(thrust::raw_pointer_cast(d_perNeuronWeights.data()),        thrust::raw_pointer_cast(perNeuronWeights.data()),          numWeights*sizeof(int), cudaMemcpyHostToDevice, stream);

    	thrust::transform(thrust::cuda::par.on(stream), d_NGN.begin(), d_NGN.end(), d_random1.begin(), d_NGN.begin(), tun1(ngnptr, numNeurons));

    	int* NGN = thrust::raw_pointer_cast(d_NGN.data());
    	int* histogram = thrust::raw_pointer_cast(d_perNeuronWeights.data());
    	auto iter = thrust::make_zip_iterator(thrust::make_tuple(d_from.begin(), d_random2.begin()));
    	thrust::transform(thrust::cuda::par.on(stream), d_weights.begin(), d_weights.end(), iter, d_weights.begin(), tun2(NGN, histogram, weightRange));

    	cudaMemcpyAsync(thrust::raw_pointer_cast(temp_weights.data()), thrust::raw_pointer_cast(d_weights.data()), numWeights*sizeof(float), cudaMemcpyHostToDevice, stream);
    }

    cusparseXcoo2csr(cusparseHandle, thrust::raw_pointer_cast(d_layer.data()), numWeights, numLayers + 2, thrust::raw_pointer_cast(d_layer_offset.data()), CUSPARSE_INDEX_BASE_ZERO);
    d_layer_offset.push_back(numWeights);

    cusparseXcoo2csr(cusparseHandle, thrust::raw_pointer_cast(d_perNeuronLayer.data()), numNeurons, numLayers + 2, thrust::raw_pointer_cast(d_perNeuronLayer_offset.data()), CUSPARSE_INDEX_BASE_ZERO);
    d_perNeuronLayer_offset.push_back(numNeurons);

    for(int j = 0; j < num_iteration; j++)
    {
		for (int i = 0; i < numLayers + 2; i++)
		{
			size_t layer_size = d_layer_offset[i+1] - d_layer_offset[i];
			int offset = d_layer_offset[i];
			if (layer_size == 0)
			{
				continue;
			}

			thrust::device_vector<float> cusparse_layer_weights(layer_size);
			thrust::device_vector<int> csr_to(layer_size);

			// cuSPARSE
			size_t pBufferSizeInBytes = 0;
			void *pBuffer = NULL;
			int* Permutation;

			cusparseXcoosort_bufferSizeExt(cusparseHandle, numNeurons, numNeurons, layer_size, thrust::raw_pointer_cast(&d_to[offset]), thrust::raw_pointer_cast(&d_from[offset]), &pBufferSizeInBytes);
			cudaMalloc(&pBuffer, sizeof(char)* pBufferSizeInBytes);
			cudaMalloc(&Permutation, sizeof(int)*layer_size);

			cusparseCreateIdentityPermutation(cusparseHandle, layer_size, Permutation);

			cusparseXcoosortByRow(cusparseHandle, numNeurons, numNeurons, layer_size, thrust::raw_pointer_cast(&d_to[offset]), thrust::raw_pointer_cast(&d_from[offset]), Permutation, pBuffer);
			cusparseSgthr(cusparseHandle, layer_size, thrust::raw_pointer_cast(&d_weights[offset]), thrust::raw_pointer_cast(cusparse_layer_weights.data()), Permutation, CUSPARSE_INDEX_BASE_ZERO);

			cudaFree(pBuffer);
			cudaFree(Permutation);

			cusparseXcoo2csr(cusparseHandle, thrust::raw_pointer_cast(&d_to[offset]), layer_size, numNeurons, thrust::raw_pointer_cast(csr_to.data()), CUSPARSE_INDEX_BASE_ZERO);

			float alpha = 1.0;
			float beta = 0.0;
			thrust::device_vector<float> y(numNeurons);
			cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, numNeurons, numNeurons, layer_size, &alpha, descr, thrust::raw_pointer_cast(cusparse_layer_weights.data()), thrust::raw_pointer_cast(csr_to.data()), thrust::raw_pointer_cast(&d_from[offset]), thrust::raw_pointer_cast(d_values.data()), &beta, thrust::raw_pointer_cast(y.data()));

			int offset_s = d_perNeuronLayer_offset[i];
			int offset_e = d_perNeuronLayer_offset[i+1];

			thrust::device_vector<float> z(numNeurons);

			thrust::gather(thrust::cuda::par.on(stream), d_neuronsID.begin(), d_neuronsID.end(), y.begin(), z.begin());

			thrust::transform(thrust::cuda::par.on(stream), &z[offset_s], &z[offset_e], &d_bias[offset_s], &z[offset_s], thrust::plus<float>());

			if (i == 0)
			{
				thrust::transform(thrust::cuda::par.on(stream), &z[offset_s], &z[offset_e], &d_values[offset_s], &z[offset_s], thrust::plus<float>());
			}

			thrust::transform(thrust::cuda::par.on(stream), &z[offset_s], &z[offset_e], &d_activationFunc[offset_s], &z[offset_s], af::activate());

			thrust::transform(thrust::cuda::par.on(stream), &d_values[offset_s], &d_values[offset_e], &z[offset_s], &d_values[offset_s], replace());
		}
    }
    cudaMemcpyAsync(thrust::raw_pointer_cast(values.data()), thrust::raw_pointer_cast(d_values.data()), numNeurons*sizeof(float), cudaMemcpyDeviceToHost, stream);
}


void nn::gen_rand_num(cudaStream_t &stream, int size)
{
    size_t n = size*numNeurons;
    size_t m = size*numWeights;
    curandGenerator_t gen;
    float *devData1, *hostData1, *devData2, *hostData2;

    /* Allocate n floats on host */
    hostData1 = (float *)calloc(n, sizeof(float));
    hostData2 = (float *)calloc(m, sizeof(float));

    /* Allocate n floats on device */
    cudaMalloc((void **)&devData1, n*sizeof(float));
    cudaMalloc((void **)&devData2, m*sizeof(float));

    /* Create pseudo-random number generator */
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    /* set stream */
    curandSetStream(gen, stream);

    /* Set seed */
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

    /* Generate n floats on device */
    curandGenerateUniform(gen, devData1, n);
    curandGenerateUniform(gen, devData2, m);

    /* Copy device memory to host */
    cudaMemcpy(hostData1, devData1, n * sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(hostData2, devData2, m * sizeof(float),cudaMemcpyDeviceToHost);

    random1.assign(hostData1, hostData1 + n);
    random2.assign(hostData2, hostData2 + m);

    /* Cleanup */
    curandDestroyGenerator(gen);
    cudaFree(devData1);
    cudaFree(devData2);
    free(hostData1);
    free(hostData2);
}


struct insert_layer
{
	int position;
	insert_layer(int _position)
	{
		position = _position;
	}

	__host__ __device__
	int operator()(int& input)
	{
		if (input > position)
		{
			return input + 1;
		}
		else
		{
			return input;
		}
	}
};

typedef thrust::tuple<int, int> inttuple;
struct checkaddweight
{
	int a;
	int b;
	checkaddweight(int _a, int _b)
	{
		a = _a;
		b = _b;
	}

	__host__ __device__
	int operator()(const inttuple& tuple)
	{
		if (a == thrust::get<0>(tuple) && b == thrust::get<1>(tuple))
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}
};

void nn::mutateTopology()
{
	currentGeneration += 1;
	std::srand(time(NULL));
	int len = std::rand()%(static_cast<int>(std::sqrt(numNeurons)));
	int* mo = new int[len];
	for (int i = 0; i < len; i++)
	{
		mo[i] = std::rand()%5;
	}
	for (int i = 0; i < len; i++)
	{
		switch (mo[i])
		{
		case 0:
		{
			neuronsID.push_back(numNeurons + 1);
			generation.push_back(currentGeneration);
			// put in new layer or not
			int det = std::rand()%numLayers;
			int l;
			if (det == 0)
			{
				int position = std::rand()%(numLayers + 2 - 1);
				l = position;
				perNeuronLayer.push_back(position + 1);
				layer.push_back(position + 1);
				thrust::transform(perNeuronLayer.begin(), perNeuronLayer.end(), perNeuronLayer.begin(), insert_layer(position));
				thrust::transform(layer.begin(), layer.end(), layer.begin(), insert_layer(position));
				numLayers += 1;
			}
			else
			{
				int position = std::rand()%(numLayers);
				l = position;
				perNeuronLayer.push_back(position);
				layer.push_back(position);
			}
			bool p1 = true;
			bool p2 = true;
			int a;
			int b;
			while(p1)
			{
				a = std::rand()%numNeurons;
				if (perNeuronLayer[a] < l)
				{
					p1 = false;
				}
			}
			while(p2)
			{
				b = std::rand()%numNeurons;
				if (perNeuronLayer[b] > l)
				{
					p2 = false;
				}
			}
			from.push_back(a);
			to.push_back(numNeurons + 1);
			from.push_back(numNeurons + 1);
			to.push_back(b);
			layer.push_back(perNeuronLayer[b]);
			float rand1 = (std::rand()*2*weightRange)/RAND_MAX - weightRange;
			float rand2 = (std::rand()*2*weightRange)/RAND_MAX - weightRange;
			weights.push_back(rand1);
			weights.push_back(rand2);
			numWeights += 2;
			numNeurons += 1;
			break;
		}
		case 1:
		{
			int a = std::rand()%numNeurons;
			int b = std::rand()%numNeurons;
			int count = 0;
			thrust::host_vector<int> pred(numWeights);
			thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(to.begin(), from.begin())), thrust::make_zip_iterator(thrust::make_tuple(to.end(), from.end())), pred.begin(), checkaddweight(a, b));
			int p = thrust::count(pred.begin(), pred.end(), 1);
			while (p == 1 && count < 20)
			{
				count += 1;
				a = std::rand()%numNeurons;
				b = std::rand()%numNeurons;
				thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(to.begin(), from.begin())), thrust::make_zip_iterator(thrust::make_tuple(to.end(), from.end())), pred.begin(), checkaddweight(a, b));
				p = thrust::count(pred.begin(), pred.end(), 1);
			}
			if (p == 0)
			{
				int w = (std::rand()*2*weightRange)/RAND_MAX - weightRange;
				weights.push_back(w);
				to.push_back(a);
				from.push_back(b);
				generation[a] = currentGeneration;
				generation[b] = currentGeneration;
				numWeights += 1;
			}
			break;
		}
		case 2:
		{
			int connection = std::rand()%numWeights;
			int count = 0;
			int a;
			int b;
			int la;
			int lb;
			while (perNeuronLayer[to[connection]] < perNeuronLayer[from[connection]] + 1 && count < 20)
			{
				connection = std::rand()%numWeights;
				count++;
			}
			a = from[connection];
			b = to[connection];
			la = perNeuronLayer[a];
			lb = perNeuronLayer[b];
			int lc = la + std::rand()%(lb - la - 1) + 1;
			neuronsID.push_back(numNeurons + 1);
			generation.push_back(currentGeneration);
			generation[a] = currentGeneration;
			generation[b] = currentGeneration;
			numNeurons += 1;
			perNeuronLayer.push_back(lc);
			to.push_back(numNeurons);
			to.push_back(b);
			from.push_back(a);
			from.push_back(numNeurons);
			layer.push_back(lc);
			layer.push_back(lb);
			float rand1 = (std::rand()*2*weightRange)/RAND_MAX - weightRange;
			float rand2 = (std::rand()*2*weightRange)/RAND_MAX - weightRange;
			weights.push_back(rand1);
			weights.push_back(rand2);
			numWeights += 2;
			break;
		}
		case 3:
		{
			int neuron = std::rand()%numNeurons;
			float _bias = (std::rand()*2*biasRange)/RAND_MAX - biasRange;
			bias[neuron] = _bias;
			generation[neuron] = currentGeneration;
			break;
		}
		case 4:
		{
			int neuron = std::rand()%numNeurons;
			int activation = std::rand()%numActivationFunc;
			activationFunc[neuron] = activation;
			generation[neuron] = currentGeneration;
			break;
		}
		}
	}
	// add neuron
	// add weight
	// splice neuron
	// add bias
	// change activation
}


