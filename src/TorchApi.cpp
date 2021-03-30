#include "TorchApi.h"
#include "TorchApiInternal.hpp"
#include "NewStuff.hpp"
#include "TensorPacking.hpp"

namespace Scripting
{

template<> constexpr const char* HandleName<Conv2D>() { return "Conv2D"; }
template<> constexpr const char* HandleName<Tensor>() { return "Tensor"; }
template<> constexpr const char* HandleName<LinearLayer>() { return "LinearLayer"; }

thread_local ScriptingState g_scriptState;

ScriptingState* GetScriptingState()
{
    return &g_scriptState;
}

NetworkState* GetNetwork() { return g_scriptState.network; }

#define NOT_NULL(name_) { if (name_ == nullptr) throw StrifeML::StrifeException("Parameter " + std::string(#name_) + " is NULL"); }

Conv2D conv2d_new(const char* name, int a, int b, int c)
{
    NOT_NULL(name);

    auto network = GetNetwork();
    auto [conv, handle] = network->conv2d.Create(name);
    conv->conv2d = network->network->module->register_module(name, torch::nn::Conv2d { a, b, c });
    return handle;
}

Conv2D conv2d_get(const char* name)
{
    NOT_NULL(name);
    return GetNetwork()->conv2d.GetHandleByName(name);
}

#define CONV2D_MEMBER_FUNCTION(name_, memberFunction_) void name_(Conv2D conv, Tensor input, Tensor output) \
{                                                                                                           \
    auto convImpl = GetNetwork()->conv2d.Get(conv);                                                         \
    auto tensorInput = g_scriptState.tensors.Get(input);                                                    \
    auto tensorOutput = g_scriptState.tensors.Get(output);                                                  \
    tensorOutput->tensor = convImpl->conv2d->memberFunction_(tensorInput->tensor);                                  \
}

#define TENSOR_FUNCTION(name_, torchName_) void name_(Tensor input) \
{                                                                                  \
    auto tensorInput = g_scriptState.tensors.Get(input);                           \
    tensorInput->tensor = torch::torchName_(tensorInput->tensor);                 \
}

#define TENSOR_MEMBER_FUNCTION(name_, memberName_) void name_(Tensor input) \
{                                                                                  \
    auto tensorInput = g_scriptState.tensors.Get(input);                           \
    tensorInput->tensor = tensorInput->tensor.memberName_();                 \
}

CONV2D_MEMBER_FUNCTION(conv2d_forward, forward)

Tensor tensor_new()
{
    auto [obj, handle] = g_scriptState.tensors.Create();
    return handle;
}

Tensor tensor_clone(Tensor input)
{
    auto [obj, handle] = g_scriptState.tensors.Create(g_scriptState.tensors.Get(input)->tensor);
    return handle;
}

Tensor tensor_new_4d(int x, int y, int z, int w)
{
    auto [obj, handle] = g_scriptState.tensors.Create(torch::IntArrayRef { x, y, z, w });
    return handle;
}

TENSOR_FUNCTION(relu, relu)

TENSOR_MEMBER_FUNCTION(tensor_squeeze, squeeze)

LinearLayer linearlayer_new(const char* name, int totalFeatures, int hiddenNodesCount)
{
    auto network = GetNetwork();
    auto [obj, handle] = network->linearLayer.Create(name);
    network->network->module->register_module(name, torch::nn::Linear(totalFeatures, hiddenNodesCount));
    return handle;
}

LinearLayer linearlayer_get(const char* name)
{
    return GetNetwork()->linearLayer.GetHandleByName(name);
}

void linearlayer_forward(LinearLayer layer, Tensor input)
{
    auto layerImpl = GetNetwork()->linearLayer.Get(layer);
    auto tensorInput = g_scriptState.tensors.Get(input);
    tensorInput->tensor = layerImpl->linear->forward(tensorInput->tensor);
}


template<typename T>
T GetProperty(StrifeML::ObjectSerializer& serializer, const char* name)
{
    auto it = serializer.schema->propertiesByName.find(name);
    if (it == serializer.schema->propertiesByName.end())
    {
        throw StrifeML::StrifeException("No such input property: %s", name);
    }
    else
    {
        auto expectedType = StrifeML::ObjectSerializerName<T>();
        if (strcmp(it->second.type, expectedType) != 0)
        {
            throw StrifeML::StrifeException(
                "%s is of type %s, but expected %s",
                name,
                it->second.type,
                expectedType);
        }

        T result;
        serializer.Seek(it->second.offset);
        StrifeML::Serializer<T>::Serialize(result, serializer);
        return result;
    }
}

Tensor pack_into_tensor(void (*selector)(Object input, Value output))
{
    auto& scriptState = g_scriptState;
    auto& valueStack = scriptState.valueStack;

    int valueHandle = valueStack.Push();
    Value value;
    value.handle = valueHandle;

    Object input;
    input.handle = valueStack.Push();

    auto inputObject = std::make_unique<ObjectImpl>();
    valueStack.GetById(input.handle).value = std::move(inputObject);

    auto doSelectorCall = [&](const SerializedInput& serializedInput) -> ValueImpl&
    {
        selector(input, value);
        return valueStack.GetById(valueHandle);
    };

    auto& tensorPackType = doSelectorCall(scriptState.network->input[0][0]).value;

    auto packedResult = std::visit([&](auto&& arg) -> torch::Tensor {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, float>)
        {
            return StrifeML::PackIntoTensor(scriptState.network->input, [=](auto& input) { return std::get<float>(doSelectorCall(input).value); });
        }
        else if constexpr (std::is_same_v<T, std::vector<float>>)
        {
            return StrifeML::PackIntoTensor(scriptState.network->input, [&](auto& input)
            {
                auto& result = std::get<std::vector<float>>(doSelectorCall(input).value);
                return gsl::span<float>(result.data(), result.size());
            });
        }
        else
        {
            throw StrifeML::StrifeException("Unhandled pack_into_tensor visitor");
        }
    }, tensorPackType);

    valueStack.Pop();
    valueStack.Pop();

    auto [obj, handle] = scriptState.tensors.Create(packedResult);
    return handle;
}

float object_get_float(Object object, const char* name)
{
    // TODO
    return 0;
}

ValueImpl& GetValue(Value value)
{
    return g_scriptState.valueStack.GetById(value.handle);
}

void value_set_float(Value value, float v)
{
    GetValue(value).value = v;
}

void value_set_float_array(Value value, float* array, int count)
{
    auto& valueImpl = GetValue(value);

    // If already a vector of floats, just resize and copy
    if (auto values = std::get_if<std::vector<float>>(&valueImpl.value))
    {
        values->resize(count);
        std::copy(array, array + count, values->data());
    }
    else
    {
        valueImpl.value = std::vector<float>(array, array + count);
    }
}

Optimizer optimizer_new_adam(const char* name, float learningRate)
{
    // TODO
    return Optimizer();
}

}