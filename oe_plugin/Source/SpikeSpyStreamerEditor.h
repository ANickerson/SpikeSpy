/*
    ------------------------------------------------------------------

    SpikeSpyStreamerEditor - Open Ephys GUI v1.x Processor Editor (API 10)

    Displays processor configuration controls and live bidirectional
    connection / heartbeat status between Open Ephys and SpikeSpy.

    ------------------------------------------------------------------
*/

#pragma once

#include <EditorHeaders.h>

class SpikeSpyStreamer;

class SpikeSpyStreamerEditor : public GenericEditor,
                               public Timer
{
public:
    SpikeSpyStreamerEditor (GenericProcessor* parentNode);
    virtual ~SpikeSpyStreamerEditor();

    void timerCallback() override;

private:
    SpikeSpyStreamer* streamer = nullptr;

    std::unique_ptr<Label> statusTitleLabel;
    std::unique_ptr<Label> statusBadge;
    std::unique_ptr<Label> detailsLabel;
    std::unique_ptr<Label> sweepsLabel;
    std::unique_ptr<Label> activityLabel;

    bool lastConnectedState = false;
    int lastSweepsCount = -1;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SpikeSpyStreamerEditor)
};
