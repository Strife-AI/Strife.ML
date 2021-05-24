#pragma once
#include <unordered_map>

namespace torch::nn
{
    class Module;
}

namespace StrifeML
{
    struct ISerializable;
    struct ObjectSerializer;

    std::shared_ptr<torch::nn::Module> CreateModule();
    void TorchLoad(std::shared_ptr<torch::nn::Module> module, std::stringstream& stream);
    void TorchSave(std::shared_ptr<torch::nn::Module> module, std::stringstream& stream);

    template<typename T>
    const char* ObjectSerializerName() { return "unknown"; };

    template<> inline const char* ObjectSerializerName<float>() { return "float"; }
    template<> inline const char* ObjectSerializerName<int>() { return "int"; }
    template<> inline const char* ObjectSerializerName<bool>() { return "bool"; }

    struct ObjectSerializerProperty
    {
        ObjectSerializerProperty()
            : type(nullptr),
              offset(0)
        {

        }

        ObjectSerializerProperty(const char* type, int offset)
            : type(type),
              offset(offset)
        {

        }

        const char* type;
        int offset;
    };

    struct ObjectSerializerSchema
    {
        template<typename T>
        void AddProperty(const char* name, int offset)
        {
            propertiesByName[name] = ObjectSerializerProperty(ObjectSerializerName<T>(), offset);
        }

        std::unordered_map<std::string, ObjectSerializerProperty> propertiesByName;
    };

    template<typename T, typename Enable = void>
    struct Serializer;

    struct ObjectSerializer
    {
        ObjectSerializer(std::vector<unsigned char>& bytes_, bool isReading_, ObjectSerializerSchema* schema = nullptr)
            : bytes(bytes_),
              isReading(isReading_),
              schema(schema)
        {

        }

        template<typename T>
        ObjectSerializer& Add(T& value, const char* name)
        {
            static_assert(!std::is_enum_v<T>, "Use AddEnum for enumerations instead of Add");

            if (schema != nullptr)
            {
                schema->template AddProperty<T>(name, (int)bytes.size());
            }

            Serializer<T>::Serialize(value, *this);
            return *this;
        }

        template<typename T>
        ObjectSerializer& AddEnum(T& value, const char* name);

        void AddBytes(unsigned char* data, int size);

        template<typename T>
        void AddBytes(T* data, int count)
        {
            AddBytes(reinterpret_cast<unsigned char*>(data), count * sizeof(T));
        }

        void Seek(int offset)
        {
            if (offset < 0 || offset >= bytes.size())
            {
                throw StrifeException("Invalid read offset");
            }

            readOffset = offset;
        }

        std::vector<unsigned char>& bytes;
        ObjectSerializerSchema* schema = nullptr;

        bool isReading;
        int readOffset = 0;
        bool hadError = false;
    };

    struct SerializedObject
    {
        template<typename T>
        void Deserialize(T& outResult);

        std::vector<unsigned char> bytes;
    };

    template <typename T>
    void SerializedObject::Deserialize(T& outResult)
    {
        static_assert(std::is_base_of_v<ISerializable, T>, "Deserialized type must implement ISerializable");
        ObjectSerializer serializer(bytes, true);
        outResult.Serialize(serializer);

        // TODO assert all bytes are used?
        // TODO check if hadError flag was set
    }

    struct ISerializable
    {
        virtual ~ISerializable() = default;

        virtual void Serialize(ObjectSerializer& serializer) = 0;
    };

    template<typename T>
    struct Serializer<T, std::enable_if_t<std::is_arithmetic_v<T>>>
    {
        static void Serialize(T& value, ObjectSerializer& serializer)
        {
            serializer.AddBytes(reinterpret_cast<unsigned char*>(&value), sizeof(value));
        }
    };

    template<typename T>
    ObjectSerializer& ObjectSerializer::AddEnum(T& value, const char* name)
    {
        int serializedValue = (int)value;

        if (schema != nullptr)
        {
            schema->template AddProperty<int>(name, (int)bytes.size());
        }

        Serializer<int>::Serialize(serializedValue, *this);

        if (isReading)
        {
            value = (T)serializedValue;
        }

        return *this;
    }
}