#pragma once

namespace StrifeML
{
    struct IDecider
    {
        virtual ~IDecider() = default;
    };

    template<typename TNeuralNetwork>
    struct Decider : IDecider
    {
        using InputType = typename TNeuralNetwork::InputType;
        using OutputType = typename TNeuralNetwork::OutputType;
        using NetworkType = TNeuralNetwork;

        Decider()
        {
            static_assert(std::is_base_of_v<INeuralNetwork, TNeuralNetwork>, "Neural network must inherit from INeuralNetwork<>");
        }

        auto MakeDecision(MlUtil::SharedArray<InputType> input, MlUtil::SharedArray<OutputType> output, int sequenceLength, int batchSize);

        std::shared_ptr<TNeuralNetwork> network = std::make_shared<TNeuralNetwork>();
        std::shared_ptr<NetworkContext<TNeuralNetwork>> networkContext;
    };

    template<typename TNetwork>
    struct MakeDecisionWorkItem : IThreadPoolWorkItem
    {
        using InputType = typename TNetwork::InputType;
        using OutputType = typename TNetwork::OutputType;

        MakeDecisionWorkItem(
            std::shared_ptr<TNetwork> network_,
            MlUtil::SharedArray<InputType> input,
            MlUtil::SharedArray<OutputType> output,
            int sequenceLength,
            int batchSize)
            : network(network_),
              input(input),
              sequenceLength(sequenceLength),
              batchSize(batchSize),
              output(output)
        {

        }

        void Execute() override
        {
            network->MakeDecision(
                Grid<const InputType>(batchSize, sequenceLength, input.data.get()),
                gsl::span<OutputType>(output.data.get(), batchSize));
        }

        std::shared_ptr<TNetwork> network;
        MlUtil::SharedArray<InputType> input;
        MlUtil::SharedArray<OutputType> output;
        int sequenceLength;
        int batchSize;
    };

    template <typename TNeuralNetwork>
    auto Decider<TNeuralNetwork>::MakeDecision(MlUtil::SharedArray<InputType> input, MlUtil::SharedArray<OutputType> output, int sequenceLength, int batchSize)
    {
        auto newNetwork = networkContext->TryGetNewNetwork();
        if (newNetwork != nullptr)
        {
            network = newNetwork;
        }

        auto workItem = std::make_shared<MakeDecisionWorkItem<TNeuralNetwork>>(network, input, output, sequenceLength, batchSize);
        auto threadPool = ThreadPool::GetInstance();
        threadPool->StartItem(workItem);
        return workItem;
    }
}