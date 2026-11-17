/*
    ------------------------------------------------------------------

    SpikeSpyStreamer - Open Ephys GUI v1.x processor plugin (API 10)

    Slices stimulus-locked ERP sweeps from the incoming continuous data on
    each rising edge of a selected TTL line, records spike latencies from
    upstream sorted units relative to the stimulus, and publishes completed
    sweeps over ZeroMQ as [JSON header, float32 waveform] multipart packets
    for SpikeSpy's live viewer. Also maintains bidirectional heartbeat
    and connection monitoring with SpikeSpy.

    ------------------------------------------------------------------
*/

#pragma once

#include <ProcessorHeaders.h>

#include <map>
#include <memory>
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
    AudioProcessorEditor* createEditor() override;

    void process (AudioBuffer<float>& continuousBuffer) override;

    void handleTTLEvent (TTLEventPtr event) override;
    void handleSpike (SpikePtr spike) override;

    String handleConfigMessage (const String& msg) override;
    void handleBroadcastMessage (const String& msg, const int64 systemTimeMillis) override;

    void parameterValueChanged (Parameter* param) override;

    void updateSettings() override;
    bool startAcquisition() override;
    bool stopAcquisition() override;

    // Connection & Diagnostics API
    bool isClientConnected() const;
    int getSweepsSent() const;
    String getClientName() const;
    int getZmqPort() const;
    int getHeartbeatPort() const;
    void pollHeartbeat (int timeoutMs);

private:
    struct Ring
    {
        std::vector<float> samples;
        int writePos = 0;
        int filled = 0;
        int64 newestSample = -1;
    };

    class HeartbeatThread;

    void bindZmq (int port);
    void closeZmq();
    void resolveSweeps (uint16 streamId);
    void publishSweep (const SlicedSweep& sweep, const float* waveform, int numSamples);
    void refreshParameterCache();

    void* zmqContext = nullptr;
    void* zmqPublisher = nullptr;
    void* zmqHeartbeat = nullptr;
    int boundZmqPort = 0;
    int boundHeartbeatPort = 0;

    std::unique_ptr<HeartbeatThread> heartbeatThread;

    mutable std::mutex stateMutex;
    std::map<uint16, Ring> rings;
    std::vector<SlicedSweep> pendingSweeps;
    bool acquiring = false;

    int cachedTtlLine = 1;
    int cachedRecChannel = 0;
    float cachedWindowMs = 500.0f;
    float cachedPreStimMs = 50.0f;
    int cachedZmqPort = 5556;

    int64 lastHeartbeatMs = 0;
    bool clientConnected = false;
    String clientName = "None";
    int totalSweepsSent = 0;
};
