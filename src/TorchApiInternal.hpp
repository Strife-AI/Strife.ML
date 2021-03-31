#pragma once

#include <variant>

#include "torch/torch.h"
#include "NewStuff.hpp"

namespace Scripting
{

    struct TensorImpl
    {
        TensorImpl()
        {

        }

        TensorImpl(const torch::IntArrayRef& ref)
            : tensor(torch::zeros(ref, torch::kFloat32))
        {

        }

        TensorImpl(const torch::Tensor& tensor)
            : tensor(tensor)
        {

        }

        torch::Tensor tensor;
    };

    struct Conv2DImpl
    {
        torch::nn::Conv2d conv2d{ nullptr };
    };

    struct OptimizerImpl
    {
        std::shared_ptr<torch::optim::Optimizer> optimizer{ nullptr };
    };

    struct LinearLayerImpl
    {
        torch::nn::Linear linear{ nullptr };
    };

    template<typename T>
    const char* HandleName();

    template<typename THandle, typename TObj>
    struct HandleMap
    {
        template<typename ... TArgs>
        std::tuple<TObj*, THandle> Create(TArgs&& ...args)
        {
            int id = objects.size();
            objects.push_back(std::make_unique<TObj>(args...));

            THandle handle;
            handle.handle = id;
            return { objects[id].get(), handle };
        }

        TObj* Get(THandle handle)
        {
            int id = handle.handle;
            if (id < 0 || id >= objects.size())
            {
                throw StrifeML::StrifeException("Invalid %s with id %d (are you using an uninitialized value?)", HandleName<THandle>(), id);
            }

            return objects[id].get();
        }

        std::vector<std::unique_ptr<TObj>> objects;
    };

    template<typename THandle, typename TObj>
    struct NamedHandleMap : private HandleMap<THandle, TObj>
    {
        template<typename ... TArgs>
        std::tuple<TObj*, THandle> Create(const char* name, TArgs ... args)
        {
            auto [obj, handle] = HandleMap<THandle, TObj>::Create(args...);
            if (objectsByName.count(name) != 0)
            {
                throw StrifeML::StrifeException("Tried to create duplicate %s: %s", HandleName<THandle>(), name);
            }

            objectsByName[name] = handle.handle;
            return { obj, handle };
        }

        THandle GetHandleByName(const char* name)
        {
            auto it = objectsByName.find(name);
            if (it == objectsByName.end())
            {
                throw StrifeML::StrifeException("No such %s: %s", HandleName<THandle>(), name);
            }
            else
            {
                THandle handle;
                handle.handle = it->second;
                return handle;
            }
        }

        TObj* Get(THandle handle)
        {
            return HandleMap<THandle, TObj>::Get(handle);
        }

        std::unordered_map<std::string, int> objectsByName;
    };

    struct SerializedInput
    {
        SerializedInput()
            : serializer(bytes, false, &schema)
        {

        }

        std::vector<unsigned char> bytes;
        StrifeML::ObjectSerializerSchema schema;
        StrifeML::ObjectSerializer serializer;
    };

    struct ObjectImpl;
    struct ValueImpl;

    struct ValueImpl
    {
        float& GetFloat()
        {
            if (auto val = std::get_if<float>(&value)) return *val;
            else throw StrifeML::StrifeException("Value is not a float");
        }

        std::variant<
            int32_t,
            int64_t,
            float,
            double,
            std::vector<float>,
            std::unique_ptr<ObjectImpl>> value;
    };

    struct ObjectImpl
    {
        std::unordered_map<std::string, ValueImpl> properties;
    };

    struct NetworkState
    {
        NetworkState(StrifeML::INeuralNetwork* network)
            : network(network)
        {

        }

        StrifeML::INeuralNetwork* network;
        NamedHandleMap<Conv2D, Conv2DImpl> conv2d;
        NamedHandleMap<Optimizer, OptimizerImpl> optimizer;
        NamedHandleMap<LinearLayer, LinearLayerImpl> linearLayer;

        Grid<SerializedInput> input;
    };

    struct ValueStack
    {
        int Push()
        {
            values.emplace_back();
            return values.size() - 1;
        }

        void Pop()
        {
            values.pop_back();
        }

        ValueImpl& GetById(int id)
        {
            if (id < 0 || id >= values.size())
            {
                throw StrifeML::StrifeException("Invalid value id: %d", id);
            }

            return values[id];
        }

        std::vector<ValueImpl> values;
    };

    struct ScriptingState
    {
        NetworkState* network = nullptr;
        HandleMap<Tensor, TensorImpl> tensors;
        ValueStack valueStack;
    };

    ScriptingState* GetScriptingState();
}