#include <vector>
#include "libtcc.h"
#include <unordered_map>

#include "Scripting.hpp"
#include "Thread/SpinLock.hpp"
#include "ScriptCompiler.hpp"

void StrifeLog(const char* format, ...)
{
    // TODO
}

static std::vector<ScriptCallableInfo*>& GetAllScriptCallableFunctions()
{
    static std::vector<ScriptCallableInfo*> functions;
    return functions;
}

bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

Script::Script(ScriptSource* source)
    : _source(source)
{

}

void ScriptCallableInfo::Initialize(const std::string_view& functionPointerPrototype)
{
    prototype = functionPointerPrototype;
    replace(prototype, "(*)", name);

    GetAllScriptCallableFunctions().push_back(this);
}

void* Script::GetSymbolOrNull(const char* name)
{
    if (_tccState == nullptr) return nullptr;
    return tcc_get_symbol(_tccState, name);
}

static void LogCompilerError(void* scriptName, const char* message)
{
    printf("Failed to compile %s: %s\n", (const char*)scriptName, message);
}

static const char* unsafeSymbols[] =
{
    "fopen",
    "gets",
    "exit",
    "longjmp",
    "setjmp",
    "abort"
};

static bool HasUnsafeSymbol(const std::string& source)
{
    for (auto symbol : unsafeSymbols)
    {
        if (source.find(symbol) != std::string::npos)
        {
            printf("Found unsafe symbol: %s\n", symbol);
            return true;
        }
    }

    return false;
}

bool Script::Compile(const char* name, const char* source)
{
    _currentScriptVersion = _source->currentVersion;

    if (_tccState != nullptr)
    {
        tcc_delete(_tccState);
    }

    _tccState = tcc_new();
    if (_tccState == nullptr)
    {
        StrifeLog("Can’t create a TCC context\n");
        return false;
    }

    tcc_set_error_func(_tccState, (void*)name, LogCompilerError);
    tcc_set_output_type(_tccState, TCC_OUTPUT_MEMORY);

    for(auto& callable : GetAllScriptCallableFunctions())
    {
        tcc_add_symbol(_tccState, callable->name, callable->functionPointer);
    }

    //tcc_set_options(_tccState, "-b");

    if (tcc_compile_string(_tccState, source) > 0
        || HasUnsafeSymbol(source))
    {
        StrifeLog("Compilation error!\n");
        tcc_delete(_tccState);
        _tccState = nullptr;
        return false;
    }

    tcc_relocate(_tccState, TCC_RELOCATE_AUTO);

    return true;
}

bool Script::TryCompile()
{
    _compilationDone = false;
    ScriptCompiler::GetInstance()->RequestCompile(shared_from_this());

    while (!_compilationDone)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return _compilationSuccessful;
}

bool Script::TryRecompileIfNewer()
{
    if (_currentScriptVersion != _source->currentVersion)
    {
        return TryCompile();
    }

    return false;
}

thread_local ThreadState threadState;

ThreadState* GetThreadState(std::thread::id threadId)
{
    return &threadState;
}

std::shared_ptr<Script> ScriptSource::CreateScript()
{
    return std::make_shared<Script>(this);
}