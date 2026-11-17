/*
    ------------------------------------------------------------------

    Open Ephys plugin entry point for the SpikeSpy Streamer processor.

    Targets the Open Ephys GUI v1.x plugin API (PLUGIN_API_VER 10).
    The getLibInfo/getPluginInfo signatures and struct layouts must match
    Source/Processors/PluginManager/OpenEphysPlugin.h in the GUI.

    ------------------------------------------------------------------
*/

#include <PluginInfo.h>

#include "SpikeSpyStreamer.h"

#ifdef _WIN32
#include <Windows.h>
#define PLUGIN_EXPORT __declspec (dllexport)
#else
#define PLUGIN_EXPORT __attribute__ ((visibility ("default")))
#endif

#define NUM_PLUGINS 1

extern "C" PLUGIN_EXPORT void getLibInfo (Plugin::LibraryInfo* info)
{
    info->apiVersion = PLUGIN_API_VER;
    info->name = "SpikeSpy";
    info->libVersion = "1.0.0";
    info->numPlugins = NUM_PLUGINS;
}

extern "C" PLUGIN_EXPORT int getPluginInfo (int index, Plugin::PluginInfo* info)
{
    switch (index)
    {
        case 0:
            info->type = Plugin::PROCESSOR;
            info->processor.name = "SpikeSpy Streamer";
            info->processor.type = Plugin::Processor::FILTER;
            info->processor.creator = &(Plugin::createProcessor<SpikeSpyStreamer>);
            return 0;
        default:
            return -1;
    }
}

#ifdef _WIN32
BOOL WINAPI DllMain (IN HINSTANCE hInst, IN DWORD reason, IN LPVOID reserved)
{
    return TRUE;
}
#endif
