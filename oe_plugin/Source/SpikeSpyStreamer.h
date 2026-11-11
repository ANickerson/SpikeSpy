/*
  SpikeSpyStreamer.h
  Open Ephys GUI v1.0+ Processor Plugin Header
*/

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <map>

// JUCE and Open Ephys minimal definitions if not compiling within Open Ephys GUI source tree
#if defined(OPEN_EPHYS_BUILD)
    #include <ProcessorHeaders.h>
#else
    // JUCE-compatible stand-in declarations for standalone header validation
    class String
    {
    public:
        String() {}
        String(const char* s) : str(s ? s : "") {}
        String(const std::string& s) : str(s) {}
        const char* toRawUTF8() const { return str.c_str(); }
        std::string toStdString() const { return str; }
        bool operator==(const String& other) const { return str == other.str; }
    private:
        std::string str;
    };

    template <typename T>
    class AudioBuffer
    {
    public:
        AudioBuffer() : numChannels(0), numSamples(0) {}
        AudioBuffer(int nChan, int nSamp) : numChannels(nChan), numSamples(nSamp)
        {
            data.resize(nChan * nSamp, 0.0f);
        }
        int getNumChannels() const { return numChannels; }
        int getNumSamples() const { return numSamples; }
        const T* getReadPointer(int chan) const { return data.data() + chan * numSamples; }
        T* getWritePointer(int chan) { return data.data() + chan * numSamples; }
        T getSample(int chan, int sample) const { return data[chan * numSamples + sample]; }
    private:
        int numChannels;
        int numSamples;
        std::vector<T> data;
    };

    struct EventChannel
    {
        int channelIndex;
        String name;
    };

    struct SpikeChannel
    {
        int channelIndex;
        String name;
    };

    struct TTLEvent
    {
        int line;
        int state;
        int64_t sampleNumber;
    };

    struct SpikeEvent
    {
        int sortedId;
        int electrodeId;
        int64_t sampleNumber;
        float peakToPeak;
    };

    class Parameter
    {
    public:
        enum Type { INT, FLOAT, BOOLEAN, STRING };
        Parameter(const String& n, Type t, double v = 0) : name(n), type(t), val(v) {}
        String name;
        Type type;
        double val;
    };

    class GenericProcessor
    {
    public:
        GenericProcessor(const String& name) : processorName(name) {}
        virtual ~GenericProcessor() {}
        virtual void process(AudioBuffer<float>& buffer) = 0;
        virtual void handleTTLEvent(const TTLEvent&) {}
        virtual void handleSpike(const SpikeEvent&) {}
        virtual void handleConfigMessage(const String&) {}
        virtual bool startAcquisition() { return true; }
        virtual bool stopAcquisition() { return true; }
        void addIntParameter(const String& name, const String& desc, int defaultVal, int minVal, int maxVal);
        void addFloatParameter(const String& name, const String& desc, float defaultVal, float minVal, float maxVal);
        void checkForEvents(bool) {}
        float getSampleRate() const { return 30000.0f; }
        std::vector<Parameter> parameters;
    private:
        String processorName;
    };
#endif

#if __has_include(<zmq.h>)
    #include <zmq.h>
#else
    // Standalone fallback declarations when compiling without libzmq development headers
    #define ZMQ_PUB 1
    #define ZMQ_LINGER 17
    #define ZMQ_SNDMORE 2
    extern "C" {
        void* zmq_ctx_new(void);
        int zmq_ctx_term(void* context);
        void* zmq_socket(void* context, int type);
        int zmq_close(void* s);
        int zmq_bind(void* s, const char* addr);
        int zmq_setsockopt(void* s, int option, const void* optval, size_t optvallen);
        int zmq_send(void* s, const void* buf, size_t len, int flags);
    }
#endif

struct SlicedSweepPending
{
    int64_t stimSample;
    int64_t stimTimestamp;
    int sweepLengthSamples;
    int preSamples;
    int recChannel;
    std::map<std::string, float> spikeLatencies;
};

/**
 * SpikeSpyStreamer Processor Plugin
 * 
 * Slices stimulus-locked sweeps in real time on incoming TTL triggers,
 * accumulates upstream spikes, and broadcasts JSON/binary ERP packets via ZeroMQ.
 * Also receives configuration updates from SpikeSpy via HTTP /api/processors/<id>/config.
 */
class SpikeSpyStreamer : public GenericProcessor
{
public:
    SpikeSpyStreamer();
    virtual ~SpikeSpyStreamer();

    void process(AudioBuffer<float>& buffer) override;
    void handleTTLEvent(const TTLEvent& event) override;
    void handleSpike(const SpikeEvent& spike) override;
    void handleConfigMessage(const String& message) override;

    bool startAcquisition() override;
    bool stopAcquisition() override;

    void setRecordingChannel(int chan);
    void setWindowMs(float ms);
    void setPreStimMs(float ms);
    void setTtlLine(int line);
    void setZmqPort(int port);

private:
    void initZmq();
    void closeZmq();
    void publishSweep(const SlicedSweepPending& sweep, const float* waveform, int numSamples);

    // ZMQ state
    void* zmqContext;
    void* zmqPublisher;
    int zmqPort;

    // Parameters
    int ttlLine;
    int recChannel;
    float windowMs;
    float preStimMs;

    // Internal circular sample buffer for real-time slicing
    int ringCapacity;
    int ringWritePos;
    int64_t totalSamplesProcessed;
    std::vector<float> ringBuffer;
    std::vector<int64_t> sampleIndices;

    std::vector<SlicedSweepPending> pendingSweeps;
    std::mutex stateMutex;
};
