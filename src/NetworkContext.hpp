#pragma once

namespace StrifeML
{
    template<typename TNeuralNetwork> struct Trainer;
    template<typename TNeuralNetwork> struct Decider;

    struct INetworkContext
    {
        virtual ~INetworkContext() = default;
    };

    template<typename TNeuralNetwork>
    struct NetworkContext : INetworkContext
    {
        NetworkContext(Decider<TNeuralNetwork>* decider_, Trainer<TNeuralNetwork>* trainer_)
            : decider(decider_),
              trainer(trainer_)
        {

        }

        virtual ~NetworkContext() = default;

        std::shared_ptr <TNeuralNetwork> SetNewNetwork(std::stringstream& stream)
        {
            newNetworkLock.Lock();
            std::shared_ptr <TNeuralNetwork> result = std::make_shared<TNeuralNetwork>();
            newNetwork = result;
            TorchLoad(newNetwork->module, stream);
            trainer->OnCreateNewNetwork(newNetwork);
            newNetworkLock.Unlock();

            return result;
        }

        std::shared_ptr <TNeuralNetwork> TryGetNewNetwork()
        {
            newNetworkLock.Lock();
            auto result = newNetwork;
            newNetwork = nullptr;
            newNetworkLock.Unlock();
            return result;
        }

        Decider <TNeuralNetwork>* decider;
        Trainer <TNeuralNetwork>* trainer;

        std::shared_ptr <TNeuralNetwork> newNetwork;
        SpinLock newNetworkLock;
        bool isEnabled = true;
        int sequenceLength;
    };
}