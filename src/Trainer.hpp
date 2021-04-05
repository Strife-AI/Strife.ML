#pragma once

#include "SampleRepository.hpp"

namespace StrifeML
{
    struct ITrainer
    {
        virtual ~ITrainer() = default;
    };

    struct TrainingBatchResult
    {
        float loss = 0;
    };

    template<typename TNeuralNetwork>
    struct RunTrainingBatchWorkItem : ThreadPoolWorkItem<TrainingBatchResult>
    {
        using InputType = typename TNeuralNetwork::InputType;
        using OutputType = typename TNeuralNetwork::OutputType;
        using NetworkType = TNeuralNetwork;
        using SampleType = Sample<InputType, OutputType>;

        RunTrainingBatchWorkItem(
            std::shared_ptr<TNeuralNetwork> network_,
            std::shared_ptr<Trainer<TNeuralNetwork>> trainer_,
            MlUtil::SharedArray<SampleType> samples_,
            int batchSize_,
            int sequenceLength_)
            : network(network_),
              samples(samples_),
              trainer(trainer_),
              batchSize(batchSize_),
              sequenceLength(sequenceLength_)
        {

        }

        void Execute() override;

        std::shared_ptr<TNeuralNetwork> network;
        MlUtil::SharedArray<SampleType> samples;
        std::shared_ptr<Trainer<TNeuralNetwork>> trainer;
        int batchSize;
        int sequenceLength;
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

        void AddSample(SampleType& sample);

        virtual void RunBatch();

        virtual bool TryCreateBatch(Grid <SampleType> outBatch);

        void NotifyTrainingComplete(std::stringstream& serializedNetwork, const TrainingBatchResult& result);

        virtual void OnTrainingComplete(const TrainingBatchResult& result) { }
        virtual void ReceiveSample(const SampleType& sample) { }

        virtual bool TrySelectSequenceSamples(gsl::span <SampleType> outSequence)
        {
            return false;
        }

        virtual void OnCreateNewNetwork(std::shared_ptr <NetworkType> newNetwork) { }

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
            RunBatch();
        }
    }

    template<typename TNeuralNetwork>
    void Trainer<TNeuralNetwork>::RunBatch()
    {
        sampleLock.Lock();
        bool successful = TryCreateBatch(Grid<SampleType>(batchSize, sequenceLength, trainingInput.data.get()));
        sampleLock.Unlock();

        if (successful)
        {
            auto taskScheduler = TaskScheduler::GetInstance();

            float runTime = trainTask != nullptr
                            ? trainTask->startTime + 1.0f / trainsPerSecond
                            : 0;

            // TODO: can save the memory allocation by reusing the task
            trainTask = std::make_shared<ScheduledTask>();
            trainTask->workItem = std::make_shared<RunTrainingBatchWorkItem<TNeuralNetwork>>(network, this->shared_from_this(), trainingInput, batchSize, sequenceLength);
            trainTask->runTime = runTime;
            taskScheduler->Start(trainTask);
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

        if (isTraining)
        {
            RunBatch();
        }
    }

    template<typename TNeuralNetwork>
    void RunTrainingBatchWorkItem<TNeuralNetwork>::Execute()
    {
        Grid<const SampleType> input(batchSize, sequenceLength, samples.data.get());
        network->TrainBatch(input, _result);
        std::stringstream stream;
        TorchSave(network->module, stream);
        trainer->NotifyTrainingComplete(stream, _result);
    }
}