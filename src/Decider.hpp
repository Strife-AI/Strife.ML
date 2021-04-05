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

        auto MakeDecision(MlUtil::SharedArray<InputType> input, int inputLength);

        std::shared_ptr<TNeuralNetwork> network = std::make_shared<TNeuralNetwork>();
        std::shared_ptr<NetworkContext<TNeuralNetwork>> networkContext;
    };

    template<typename TNetwork>
    struct MakeDecisionWorkItem : ThreadPoolWorkItem<typename TNetwork::OutputType>
    {
        using InputType = typename TNetwork::InputType;

        MakeDecisionWorkItem(std::shared_ptr<TNetwork> network_, MlUtil::SharedArray<InputType> input_, int inputLength_)
            : network(network_),
              input(input_),
              inputLength(inputLength_)
        {

        }

        void Execute() override
        {
            network->MakeDecision(Grid<const InputType>(1, inputLength, input.data.get()), this->_result);
        }

        std::shared_ptr<TNetwork> network;
        MlUtil::SharedArray<InputType> input;
        int inputLength;
    };

    template <typename TNeuralNetwork>
    auto Decider<TNeuralNetwork>::MakeDecision(MlUtil::SharedArray<InputType> input, int inputLength)
    {
        auto newNetwork = networkContext->TryGetNewNetwork();
        if (newNetwork != nullptr)
        {
            network = newNetwork;
        }

        auto workItem = std::make_shared<MakeDecisionWorkItem<TNeuralNetwork>>(network, input, inputLength);
        auto threadPool = ThreadPool::GetInstance();
        threadPool->StartItem(workItem);
        return workItem;
    }
}