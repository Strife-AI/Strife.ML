#pragma once

#include <variant>
#include <torch/torch.h>
#include "TorchApi.h"
#include "Scripting/Scripting.hpp"
#include "TorchApiInternal.hpp"

void StrifeLog(const char* format, ...);

template<typename TInput, typename TOutput>
struct ScriptNetwork : StrifeML::NeuralNetwork<TInput, TOutput, 1>
{
    using SampleType = StrifeML::Sample<TInput, TOutput>;

    void MakeDecision(Grid<const TInput> input, TOutput& output) override
    {
        VariableSizedGrid<Scripting::SerializedInput> serializedInput(input.Rows(), input.Cols());

        for (int i = 0; i < input.Rows(); ++i)
        {
            for (int j = 0; j < input.Cols(); ++j)
            {
                // Safe to remove constness because we're in writing mode
                auto inputElement = const_cast<TInput&>(input[i][j]);
                serializedInput[i][j].serializer.isReading = false;
                inputElement.Serialize(serializedInput[i][j].serializer);
            }
        }

        networkState.input = serializedInput;

        try
        {
            DoScriptCall([&]
            {
                Scripting::Value action = Scripting::PushValue();
                makeDecision(action);
                auto& actionImpl = Scripting::GetValue(action);
                OnReceiveScriptDecision(actionImpl, output);
            });
        }
        catch(...)
        {
            // TODO
        }
    }

    virtual void OnReceiveScriptDecision(Scripting::ValueImpl& scriptResult, TOutput& output) = 0;

    void TrainBatch(Grid<const SampleType> input, StrifeML::TrainingBatchResult& outResult) override
    {
        VariableSizedGrid<Scripting::SerializedInput> serializedInput(input.Rows(), input.Cols());
        PackSerializedInput(input, serializedInput);

        try
        {
            DoScriptCall([&]
            {
                Scripting::Value loss = Scripting::PushValue();
                train(loss);
                auto& lossImpl = Scripting::GetValue(loss);

                if (auto lossFloat = std::get_if<float>(&lossImpl.value))
                {
                    outResult.loss = *lossFloat;
                }
            });
        }
        catch (...)
        {
            // TODO
        }

        networkState.input.Set(0, 0, nullptr);
    }

    void PackSerializedInput(const Grid<const SampleType>& input, VariableSizedGrid<Scripting::SerializedInput>& serializedInput)
    {
        for (int i = 0; i < input.Rows(); ++i)
        {
            for (int j = 0; j < input.Cols(); ++j)
            {
                // Safe to remove constness because we're in writing mode
                auto inputElement = const_cast<SampleType&>(input[i][j]);
                serializedInput[i][j].serializer.isReading = false;
                inputElement.input.Serialize(serializedInput[i][j].serializer);
                inputElement.output.Serialize(serializedInput[i][j].serializer);
            }
        }

        networkState.input = serializedInput;
    }

    void BindCallbacks(std::shared_ptr<Script> script, bool runSetup)
    {
        if (!script->TryBindFunction(train))
        {
            StrifeLog("Failed to bind Train() function\n");
        }

        if (!script->TryBindFunction(makeDecision))
        {
            StrifeLog("Failed to bind MakeDecision() function\n");
        }

        if (script->TryBindFunction(setup) && runSetup)
        {
            DoScriptCall([=] { setup(); });
        }
    }

    template<typename TFunc>
    void DoScriptCall(TFunc func)
    {
        auto scriptState = Scripting::GetScriptingState();
        scriptState->network = &networkState;
        func();
        scriptState->tensors.objects.clear();
        scriptState->network = nullptr;
    }

    ScriptFunction<void()> setup { "Setup" };
    ScriptFunction<void(Scripting::Value outLoss)> train { "Train" };
    ScriptFunction<void(Scripting::Value outAction)> makeDecision { "MakeDecision" };
    Scripting::NetworkState networkState { this };
    bool hasScriptError = false;
};

template<typename TScriptNetwork>
struct ScriptTrainer : StrifeML::Trainer<TScriptNetwork>
{
    using TInput = typename TScriptNetwork::InputType;
    using TOutput = typename TScriptNetwork::OutputType;
    using SampleType = StrifeML::Sample<TInput, TOutput>;

    ScriptTrainer(ScriptSource* source, int batchSize, float trainsPerSecond)
        : StrifeML::Trainer<TScriptNetwork>(batchSize, trainsPerSecond),
          source(source)
    {
        this->minSamplesBeforeStartingTraining = -1;
        script = source->CreateScript();
        script->TryCompile();   // TODO error checking
    }

    void OnRunBatch() override
    {
        if (script->TryRecompileIfNewer())
        {
            StrifeLog("Successfully recompiled\n");
            this->network->BindCallbacks(script, false);
        }
    }

    void OnCreateNewNetwork(std::shared_ptr<TScriptNetwork> newNetwork)
    {
        newNetwork->BindCallbacks(script, true);
    }

    ScriptSource* source;
    std::shared_ptr<Script> script;
};

template<typename TInput, typename TOutput>
struct ScriptDecider : StrifeML::Decider<ScriptNetwork<TInput, TOutput>>
{

};