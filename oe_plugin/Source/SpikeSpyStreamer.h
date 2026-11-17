/*
    ------------------------------------------------------------------

    SpikeSpyStreamer - Open Ephys GUI v1.x processor plugin (API 10)

    Slices stimulus-locked ERP sweeps from the incoming continuous data on
    each rising edge of a selected TTL line, records spike latencies from
    upstream sorted units relative to the stimulus, and publishes completed
    sweeps over ZeroMQ as [JSON header, float32 waveform] multipart packets
    for SpikeSpy's live viewer.

    ------------------------------------------------------------------
*/

#pragma once

#include <ProcessorHeaders.h>

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <zmq.h>

struct SlicedSweep
{
    uint16 streamId = 0;
    int64 stimSample = 0;
    float sampleRate = 30000.0f;
    int sweepLengthSamples = 0;
    int preSamples = 0;
    int recChannel = 0;
    std::map<std::string, float> spikeLatencies;
};

class SpikeSpyStreamer : public GenericProcessor
{
public:
    SpikeSpyStreamer();
    virtual ~SpikeSpyStreamer();

    void registerParameters() override;

    void process (AudioBuffer<float>& continuousBuffer) override;

    void handleTTLEvent (TTLEventPtr event) override;
    void handleSpike (SpikePtr spike) override;

    String handleConfigMessage (const String& msg) override;
    void handleBroadcastMessage (const String& msg, const int64 systemTimeMillis) override;

    void parameterValueChanged (Parameter* param) override;

    void updateSettings() override;
    bool startAcquisition() override;
    bool stopAcquisition() override;

private:
    struct Ring
    {
        std::vector<float> samples;
        int writePos = 0;
        int filled = 0;
        int64 newestSample = -1;
    };

    void bindZmq (int port);
    void closeZmq();
    void resolveSweeps (uint16 streamId);
    void publishSweep (const SlicedSweep& sweep, const float* waveform, int numSamples);
    void refreshParameterCache();

    void* zmqContext = nullptr;
    void* zmqPublisher = nullptr;
    int boundZmqPort = 0;

    std::mutex stateMutex;
    std::map<uint16, Ring> rings;
    std::vector<SlicedSweep> pendingSweeps;
    bool acquiring = false;

    int cachedTtlLine = 1;
    int cachedRecChannel = 0;
    float cachedWindowMs = 500.0f;
    float cachedPreStimMs = 50.0f;
    int cachedZmqPort = 5556;
};
