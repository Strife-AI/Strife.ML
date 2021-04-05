#pragma once

namespace StrifeML
{
    namespace MlUtil
    {
        template<typename T>
        struct SharedArray
        {
            SharedArray(int count_)
                : count(count_)
            {
                data = MakeSharedArray(count);
            }

            std::shared_ptr<T> data;
            int count;

        private:
            std::shared_ptr<T> MakeSharedArray(int count)
            {
                // This *should* go away with C++ 17 since it should provide a version of std::make_shared<> for arrays, but that doesn't seem
                // to be the case in MSVC
                return std::shared_ptr<T>(new T[count], [](T* ptr)
                {
                    delete[] ptr;
                });
            }
        };
    }

    struct StrifeException : std::exception
    {
        StrifeException(const std::string& message_)
            : message(message_)
        {

        }

        StrifeException(const char* format, ...)
        {
            va_list args;
            va_start (args, format);

            char buf[1024];
            vsnprintf(buf, sizeof(buf), format, args);
            message = buf;
        }

        const char* what()  const noexcept override
        {
            return message.c_str();
        }

        std::string message;
    };

    struct RandomNumberGenerator
    {
        RandomNumberGenerator()
            : _rng(std::random_device()())
        {

        }

        int RandInt(int min, int max)
        {
            return std::uniform_int_distribution<int>(min, max)(_rng);
        }

        int RandFloat(float min, float max)
        {
            return std::uniform_real_distribution<float>(min, max)(_rng);
        }

    private:
        std::mt19937 _rng;
    };
}