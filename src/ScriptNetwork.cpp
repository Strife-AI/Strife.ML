#include "ScriptNetwork.hpp"
#include "TorchApi.h"

using namespace Scripting;

void RegisterScriptFunctions()
{
    SCRIPT_REGISTER(tensor_new);
    SCRIPT_REGISTER(tensor_new_4d);
    SCRIPT_REGISTER(tensor_clone);

    SCRIPT_REGISTER(conv2d_add)
    SCRIPT_REGISTER(conv2d_get)
    SCRIPT_REGISTER(conv2d_forward);

    SCRIPT_REGISTER(relu);

    SCRIPT_REGISTER(linearlayer_add);
    SCRIPT_REGISTER(linearlayer_get);
    SCRIPT_REGISTER(linearlayer_forward);

    SCRIPT_REGISTER(pack_float_array);
};