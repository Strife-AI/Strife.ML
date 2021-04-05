#pragma once

namespace StrifeML
{
    struct INeuralNetwork
    {
        INeuralNetwork()
            : module(CreateModule())
        {

        }

        std::shared_ptr <torch::nn::Module> module;
        virtual ~INeuralNetwork() = default;
    };

    struct TrainingBatchResult;

    template<typename TInput, typename TOutput, int SeqLength>
    struct NeuralNetwork : INeuralNetwork
    {
        using InputType = TInput;
        using OutputType = TOutput;
        using SampleType = Sample<InputType, OutputType>;
        static constexpr int SequenceLength = SeqLength;

        virtual void MakeDecision(Grid<const TInput> input, TOutput& output) = 0;
        virtual void TrainBatch(Grid<const SampleType> input, TrainingBatchResult& outResult) = 0;
    };
}