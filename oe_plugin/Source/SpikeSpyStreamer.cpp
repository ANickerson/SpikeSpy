/*
  SpikeSpyStreamer.cpp
  Implementation of Open Ephys v1.0+ SpikeSpyStreamer Processor Plugin
*/

#include "SpikeSpyStreamer.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <algorithm>

#if !defined(OPEN_EPHYS_BUILD)
void GenericProcessor::addIntParameter(const String& name, const String&, int defaultVal, int, int)
{
    parameters.push_back(Parameter(name, Parameter::INT, defaultVal));
}

void GenericProcessor::addFloatParameter(const String& name, const String&, float defaultVal, float, float)
{
    parameters.push_back(Parameter(name, Parameter::FLOAT, defaultVal));
}
#endif

SpikeSpyStreamer::SpikeSpyStreamer()
    : GenericProcessor("SpikeSpy Streamer"),
      zmqContext(nullptr),
      zmqPublisher(nullptr),
      zmqPort(5556),
      ttlLine(1),
      recChannel(0),
      windowMs(500.0f),
      preStimMs(50.0f),
      ringCapacity(30000 * 10), // 10 seconds capacity at 30 kHz
      ringWritePos(0),
      totalSamplesProcessed(0)
{
    // Register Open Ephys v1.0 parameters
    addIntParameter("ttl_line", "TTL Event Line to trigger sweeps", 1, 1, 16);
    addIntParameter("rec_channel", "Analog channel to slice for ERP", 0, 0, 512);
    addFloatParameter("window_ms", "ERP Sweep Window (ms)", 500.0f, 10.0f, 5000.0f);
    addFloatParameter("pre_stim_ms", "Pre-stimulus offset (ms)", 50.0f, 0.0f, 500.0f);
    addIntParameter("zmq_port", "ZeroMQ TCP Port to publish ERP sweeps", 5556, 1024, 65535);

    ringBuffer.resize(ringCapacity, 0.0f);
    sampleIndices.resize(ringCapacity, 0);

    initZmq();
}

SpikeSpyStreamer::~SpikeSpyStreamer()
{
    closeZmq();
}

void SpikeSpyStreamer::initZmq()
{
    std::lock_guard<std::mutex> lock(stateMutex);
    if (!zmqContext)
    {
        zmqContext = zmq_ctx_new();
    }
    if (!zmqPublisher)
    {
        zmqPublisher = zmq_socket(zmqContext, ZMQ_PUB);
        std::string endpoint = "tcp://*:" + std::to_string(zmqPort);
        int linger = 0;
        zmq_setsockopt(zmqPublisher, ZMQ_LINGER, &linger, sizeof(linger));
        zmq_bind(zmqPublisher, endpoint.c_str());
    }
}

void SpikeSpyStreamer::closeZmq()
{
    std::lock_guard<std::mutex> lock(stateMutex);
    if (zmqPublisher)
    {
        zmq_close(zmqPublisher);
        zmqPublisher = nullptr;
    }
    if (zmqContext)
    {
        zmq_ctx_term(zmqContext);
        zmqContext = nullptr;
    }
}

bool SpikeSpyStreamer::startAcquisition()
{
    std::lock_guard<std::mutex> lock(stateMutex);
    ringWritePos = 0;
    totalSamplesProcessed = 0;
    pendingSweeps.clear();
    return true;
}

bool SpikeSpyStreamer::stopAcquisition()
{
    return true;
}

void SpikeSpyStreamer::handleTTLEvent(const TTLEvent& event)
{
    // Only trigger on rising edge (state == 1) of selected TTL line
    if (event.state == 1 && event.line == ttlLine)
    {
        std::lock_guard<std::mutex> lock(stateMutex);
        float fs = getSampleRate();
        int windowSamples = static_cast<int>((windowMs / 1000.0f) * fs);
        int preSamples = static_cast<int>((preStimMs / 1000.0f) * fs);

        SlicedSweepPending sweep;
        sweep.stimSample = event.sampleNumber;
        sweep.stimTimestamp = event.sampleNumber;
        sweep.sweepLengthSamples = windowSamples;
        sweep.preSamples = preSamples;
        sweep.recChannel = recChannel;

        pendingSweeps.push_back(sweep);
        if (pendingSweeps.size() > 200)
        {
            pendingSweeps.erase(pendingSweeps.begin());
        }
    }
}

void SpikeSpyStreamer::handleSpike(const SpikeEvent& spike)
{
    std::lock_guard<std::mutex> lock(stateMutex);
    float fs = getSampleRate();
    std::string unitName = "unit_" + std::to_string(spike.sortedId);

    // Attribute spike to any pending sweeps containing this sample
    for (auto& sweep : pendingSweeps)
    {
        if (spike.sampleNumber >= sweep.stimSample &&
            spike.sampleNumber <= sweep.stimSample + sweep.sweepLengthSamples)
        {
            float latencyS = static_cast<float>(spike.sampleNumber - sweep.stimSample) / fs;
            sweep.spikeLatencies[unitName] = latencyS;
        }
    }
}

void SpikeSpyStreamer::handleConfigMessage(const String& message)
{
    std::lock_guard<std::mutex> lock(stateMutex);
    std::string text = message.toStdString();
    std::cout << "[SpikeSpyStreamer] Received config message: " << text << std::endl;

    // Parse simple commands or JSON annotations sent from SpikeSpy
    // Example: "setStimVoltage: 10.0" or JSON {"type": "unit_annotation", ...}
    if (text.find("setStimVoltage:") != std::string::npos)
    {
        // Parameter update logic or closed-loop hardware command
    }
}

void SpikeSpyStreamer::process(AudioBuffer<float>& buffer)
{
    checkForEvents(true);

    int numSamples = buffer.getNumSamples();
    int numChannels = buffer.getNumChannels();
    if (numSamples <= 0 || numChannels <= 0)
        return;

    int targetChan = std::min(std::max(0, recChannel), numChannels - 1);
    const float* src = buffer.getReadPointer(targetChan);

    std::lock_guard<std::mutex> lock(stateMutex);

    // 1. Copy incoming audio samples into circular ring buffer
    for (int i = 0; i < numSamples; ++i)
    {
        ringBuffer[ringWritePos] = src[i];
        sampleIndices[ringWritePos] = totalSamplesProcessed + i;
        ringWritePos = (ringWritePos + 1) % ringCapacity;
    }
    int64_t currentLatestSample = totalSamplesProcessed + numSamples - 1;
    totalSamplesProcessed += numSamples;

    // 2. Check pending sweeps to see if enough future samples have arrived
    auto it = pendingSweeps.begin();
    while (it != pendingSweeps.end())
    {
        int64_t startSample = it->stimSample - it->preSamples;
        int64_t endSample = it->stimSample + it->sweepLengthSamples;
        int totalSweepSamples = static_cast<int>(endSample - startSample);

        if (currentLatestSample >= endSample)
        {
            // Slicing window is fully present in ring buffer
            std::vector<float> slicedWaveform(totalSweepSamples, 0.0f);

            int64_t distFromLatest = currentLatestSample - startSample;
            int bufStartPos = (ringWritePos - 1 - distFromLatest) % ringCapacity;
            if (bufStartPos < 0) bufStartPos += ringCapacity;

            for (int s = 0; s < totalSweepSamples; ++s)
            {
                int pos = (bufStartPos + s) % ringCapacity;
                slicedWaveform[s] = ringBuffer[pos];
            }

            // Publish sliced ERP packet over ZeroMQ
            publishSweep(*it, slicedWaveform.data(), totalSweepSamples);

            it = pendingSweeps.erase(it);
        }
        else
        {
            // Sweep still waiting for future samples
            ++it;
        }
    }
}

void SpikeSpyStreamer::publishSweep(const SlicedSweepPending& sweep, const float* waveform, int numSamples)
{
    if (!zmqPublisher)
        return;

    float fs = getSampleRate();
    double stimTimestampS = static_cast<double>(sweep.stimSample) / fs;
    double preOffsetS = static_cast<double>(sweep.preSamples) / fs;

    // Build JSON metadata header
    std::ostringstream jsonStream;
    jsonStream << "{"
               << "\"type\":\"erp_sweep\","
               << "\"sample_rate\":" << fs << ","
               << "\"stim_timestamp_s\":" << std::fixed << std::setprecision(6) << stimTimestampS << ","
               << "\"pre_offset_s\":" << preOffsetS << ","
               << "\"num_samples\":" << numSamples << ","
               << "\"rec_channel\":" << sweep.recChannel << ","
               << "\"spike_latencies_s\":{";

    bool first = true;
    for (const auto& kv : sweep.spikeLatencies)
    {
        if (!first) jsonStream << ",";
        jsonStream << "\"" << kv.first << "\":" << kv.second;
        first = false;
    }
    jsonStream << "}}";

    std::string jsonStr = jsonStream.str();

    // Send multipart ZMQ message:
    // Frame 0: JSON header metadata
    // Frame 1: Binary float32 waveform buffer
    zmq_send(zmqPublisher, jsonStr.c_str(), jsonStr.size(), ZMQ_SNDMORE);
    zmq_send(zmqPublisher, waveform, numSamples * sizeof(float), 0);
}

void SpikeSpyStreamer::setRecordingChannel(int chan) { recChannel = chan; }
void SpikeSpyStreamer::setWindowMs(float ms) { windowMs = ms; }
void SpikeSpyStreamer::setPreStimMs(float ms) { preStimMs = ms; }
void SpikeSpyStreamer::setTtlLine(int line) { ttlLine = line; }
void SpikeSpyStreamer::setZmqPort(int port)
{
    if (zmqPort != port)
    {
        zmqPort = port;
        closeZmq();
        initZmq();
    }
}
