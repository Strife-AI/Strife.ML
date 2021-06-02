#pragma once

namespace StrifeML
{
    struct INeuralNetwork
    {
        INeuralNetwork()
            : module(CreateModule())
        {

        }

        virtual ~INeuralNetwork() = default;

        std::shared_ptr <torch::nn::Module> module;
    };

    struct TrainingBatchResult;

    template<typename TInput, typename TOutput>
    struct NeuralNetwork : INeuralNetwork
    {
        using InputType = TInput;
        using OutputType = TOutput;
        using SampleType = Sample<InputType, OutputType>;

        NeuralNetwork(int sequenceLength)
            : sequenceLength(sequenceLength)
        {

        }

        virtual void MakeDecision(Grid<const TInput> input, gsl::span<TOutput> output) = 0;
        virtual void TrainBatch(Grid<const SampleType> input, TrainingBatchResult& outResult) = 0;

        int sequenceLength;
    };
}