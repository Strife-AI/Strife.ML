#pragma once

#include "SampleRepository.hpp"

#include <sstream>

namespace StrifeML
{
    struct ITrainer
    {
        virtual ~ITrainer() = default;
    };

    struct TrainingBatchResult
    {
        float loss = 0;
        bool isSuccess = true;
    };

    template<typename TNeuralNetwork>
    struct RunTrainingBatchWorkItem : ThreadPoolWorkItem<TrainingBatchResult>
    {
        using InputType = typename TNeuralNetwork::InputType;
        using OutputType = typename TNeuralNetwork::OutputType;
        using NetworkType = TNeuralNetwork;
        using SampleType = Sample<InputType, OutputType>;

        RunTrainingBatchWorkItem(std::shared_ptr<Trainer<TNeuralNetwork>> trainer)
            : trainer(trainer)
        {

        }

        void Execute() override;

        std::shared_ptr<Trainer<TNeuralNetwork>> trainer;
    };

    template<typename TNeuralNetwork>
    struct Trainer : ITrainer, std::enable_shared_from_this<Trainer<TNeuralNetwork>>
    {
        using InputType = typename TNeuralNetwork::InputType;
        using OutputType = typename TNeuralNetwork::OutputType;
        using NetworkType = TNeuralNetwork;
        using SampleType = Sample<InputType, OutputType>;

        static_assert(std::is_base_of_v < INeuralNetwork, TNeuralNetwork > , "Neural network must inherit from INeuralNetwork<>");

        Trainer(int batchSize_, float trainsPerSecond_);

        void StartRunning();

        void AddSample(SampleType& sample);

        virtual bool TryCreateBatch(Grid <SampleType> outBatch);

        void NotifyTrainingComplete(std::stringstream& serializedNetwork, const TrainingBatchResult& result);

        virtual void OnTrainingComplete(const TrainingBatchResult& result) { }
        virtual void ReceiveSample(const SampleType& sample) { }

        virtual bool TrySelectSequenceSamples(gsl::span <SampleType> outSequence)
        {
            return false;
        }

        virtual void OnCreateNewNetwork(std::shared_ptr <NetworkType> newNetwork) { }
        virtual void OnRunBatch() { }

        SpinLock sampleLock;
        RandomNumberGenerator rng;
        SampleRepository <SampleType> sampleRepository;
        MlUtil::SharedArray <SampleType> trainingInput;
        int batchSize;
        int sequenceLength;
        float trainsPerSecond;
        std::shared_ptr <ScheduledTask> trainTask;
        std::shared_ptr <NetworkContext<TNeuralNetwork>> networkContext;
        std::shared_ptr <TNeuralNetwork> network;
        bool isTraining = false;
        int minSamplesBeforeStartingTraining = 32;
        int totalSamples = 0;
    };

    template<typename TNeuralNetwork>
    Trainer<TNeuralNetwork>::Trainer(int batchSize_, float trainsPerSecond_)
        : sampleRepository(rng),
          trainingInput(MlUtil::SharedArray<SampleType>(batchSize_ * TNeuralNetwork::SequenceLength)),
          batchSize(batchSize_),
          sequenceLength(TNeuralNetwork::SequenceLength),
          trainsPerSecond(trainsPerSecond_)
    {

    }

    template<typename TNeuralNetwork>
    void Trainer<TNeuralNetwork>::AddSample(Trainer::SampleType& sample)
    {
        sampleLock.Lock();
        ReceiveSample(sample);
        sampleLock.Unlock();

        ++totalSamples;
        if (!isTraining && totalSamples >= minSamplesBeforeStartingTraining)
        {
            isTraining = true;
        }
    }

    template<typename TNeuralNetwork>
    bool Trainer<TNeuralNetwork>::TryCreateBatch(Grid <SampleType> outBatch)
    {
        int batchSize = outBatch.Rows();
        for (int i = 0; i < batchSize; ++i)
        {
            if (!TrySelectSequenceSamples(gsl::span<SampleType>(outBatch[i], outBatch.Cols())))
            {
                return false;
            }
        }

        return true;
    }

    template<typename TNeuralNetwork>
    void Trainer<TNeuralNetwork>::NotifyTrainingComplete(std::stringstream& serializedNetwork, const TrainingBatchResult& result)
    {
        auto newNetwork = networkContext->SetNewNetwork(serializedNetwork);
        OnTrainingComplete(result);
    }

    template<typename TNeuralNetwork>
    void Trainer<TNeuralNetwork>::StartRunning()
    {
        auto taskScheduler = TaskScheduler::GetInstance();
        trainTask = std::make_shared<ScheduledTask>();
        trainTask->workItem = std::make_shared<RunTrainingBatchWorkItem<TNeuralNetwork>>(this->shared_from_this());
        trainTask->recurringTime = 1.0f / trainsPerSecond;
        trainTask->runTime = 0;
        taskScheduler->Start(trainTask);
    }

    template<typename TNeuralNetwork>
    void RunTrainingBatchWorkItem<TNeuralNetwork>::Execute()
    {
        if (!trainer->isTraining)
        {
            return;
        }

        trainer->sampleLock.Lock();
        bool successful = trainer->TryCreateBatch(Grid<SampleType>(trainer->batchSize, trainer->sequenceLength, trainer->trainingInput.data.get()));
        trainer->sampleLock.Unlock();

        if (successful)
        {
            trainer->OnRunBatch();
            Grid<const SampleType> input(trainer->batchSize, trainer->sequenceLength, trainer->trainingInput.data.get());
            trainer->network->TrainBatch(input, _result);
            std::stringstream stream;
            TorchSave(trainer->network->module, stream);
            trainer->NotifyTrainingComplete(stream, _result);
        }
    }
}