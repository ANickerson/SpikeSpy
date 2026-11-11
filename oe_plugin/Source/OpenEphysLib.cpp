/*
  Open Ephys Plugin Library Interface for SpikeSpyStreamer
  Open Ephys GUI v1.0+ Plugin API (PLUGIN_API_VER = 10)
*/

#include "SpikeSpyStreamer.h"
#include <string>

#ifndef PLUGIN_API_VER
#define PLUGIN_API_VER 10
#endif

namespace Plugin
{
    enum Type
    {
        PROCESSOR = 0,
        RECORD_ENGINE,
        DATA_THREAD,
        FILE_SOURCE
    };

    struct PluginInfo
    {
        Type type;
        const char* name;
        const char* description;
        const char* creator;
        float version;
        void* (*creatorFunction)();
    };

    struct LibraryInfo
    {
        int apiVersion;
        const char* name;
        const char* version;
        int numPlugins;
    };
}

static void* createSpikeSpyStreamer()
{
    return new SpikeSpyStreamer();
}

extern "C"
{
    /**
     * Called by Open Ephys PluginManager to verify API compatibility
     */
    void getLibInfo(Plugin::LibraryInfo* info)
    {
        info->apiVersion = PLUGIN_API_VER;
        info->name = "SpikeSpy Streamer Plugin";
        info->version = "1.0.0";
        info->numPlugins = 1;
    }

    /**
     * Returns metadata and creator function for the SpikeSpyStreamer processor plugin
     */
    int getPluginInfo(int index, Plugin::PluginInfo* info)
    {
        if (index == 0)
        {
            info->type = Plugin::PROCESSOR;
            info->name = "SpikeSpy Streamer";
            info->description = "Real-time stimulus-locked ERP slicer and ZeroMQ streamer for SpikeSpy";
            info->creator = "SpikeSpy Team";
            info->version = 1.0f;
            info->creatorFunction = &createSpikeSpyStreamer;
            return 0;
        }
        return -1;
    }
}
