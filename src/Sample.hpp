#pragma once

namespace StrifeML
{
    template<typename TInput, typename TOutput>
    struct Sample
    {
        TInput input;
        TOutput output;
    };
}