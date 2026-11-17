/*
    ------------------------------------------------------------------

    SpikeSpyStreamer - Open Ephys GUI v1.x processor plugin (API 10)

    ------------------------------------------------------------------
*/

#include "SpikeSpyStreamer.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace
{
const int RING_SECONDS = 10;
const int MAX_PENDING_SWEEPS = 200;
} // namespace

SpikeSpyStreamer::SpikeSpyStreamer()
    : GenericProcessor ("SpikeSpy Streamer")
{
}

SpikeSpyStreamer::~SpikeSpyStreamer()
{
    std::lock_guard<std::mutex> lock (stateMutex);
    closeZmq();
}

void SpikeSpyStreamer::registerParameters()
{
    addIntParameter (Parameter::PROCESSOR_SCOPE,
                     "ttl_line",
                     "TTL Line",
                     "TTL event line (1-based) whose rising edge triggers an ERP sweep",
                     1,
                     1,
                     16);

    addIntParameter (Parameter::PROCESSOR_SCOPE,
                     "rec_channel",
                     "Record Channel",
                     "Continuous channel (local index within each stream) to slice sweeps from",
                     0,
                     0,
                     512);

    addFloatParameter (Parameter::PROCESSOR_SCOPE,
                       "window_ms",
                       "Window (ms)",
                       "Total ERP sweep window after the stimulus",
                       "ms",
                       500.0f,
                       10.0f,
                       5000.0f,
                       10.0f);

    addFloatParameter (Parameter::PROCESSOR_SCOPE,
                       "pre_stim_ms",
                       "Pre-stim (ms)",
                       "Pre-stimulus time included at the start of each sweep",
                       "ms",
                       50.0f,
                       0.0f,
                       500.0f,
                       1.0f);

    addIntParameter (Parameter::PROCESSOR_SCOPE,
                     "zmq_port",
                     "ZMQ Port",
                     "TCP port for the ZeroMQ ERP sweep publisher",
                     5556,
                     1024,
                     65535);
}

void SpikeSpyStreamer::refreshParameterCache()
{
    cachedTtlLine = (int) getParameter ("ttl_line")->getValue();
    cachedRecChannel = (int) getParameter ("rec_channel")->getValue();
    cachedWindowMs = (float) getParameter ("window_ms")->getValue();
    cachedPreStimMs = (float) getParameter ("pre_stim_ms")->getValue();
    cachedZmqPort = (int) getParameter ("zmq_port")->getValue();
}

void SpikeSpyStreamer::bindZmq (int port)
{
    closeZmq();

    zmqContext = zmq_ctx_new();
    if (zmqContext == nullptr)
    {
        DBG ("SpikeSpyStreamer: failed to create ZMQ context");
        return;
    }

    zmqPublisher = zmq_socket (zmqContext, ZMQ_PUB);
    if (zmqPublisher == nullptr)
    {
        DBG ("SpikeSpyStreamer: failed to create ZMQ publisher socket");
        zmq_ctx_term (zmqContext);
        zmqContext = nullptr;
        return;
    }

    int linger = 0;
    zmq_setsockopt (zmqPublisher, ZMQ_LINGER, &linger, sizeof (linger));

    int sndHwm = 30;
    zmq_setsockopt (zmqPublisher, ZMQ_SNDHWM, &sndHwm, sizeof (sndHwm));

    const std::string endpoint = "tcp://*:" + std::to_string (port);
    if (zmq_bind (zmqPublisher, endpoint.c_str()) != 0)
    {
        DBG ("SpikeSpyStreamer: zmq_bind failed on " << endpoint);
        zmq_close (zmqPublisher);
        zmqPublisher = nullptr;
        zmq_ctx_term (zmqContext);
        zmqContext = nullptr;
        return;
    }

    boundZmqPort = port;
    DBG ("SpikeSpyStreamer: ZMQ publisher bound to " << endpoint);
}

void SpikeSpyStreamer::closeZmq()
{
    if (zmqPublisher != nullptr)
    {
        zmq_close (zmqPublisher);
        zmqPublisher = nullptr;
    }

    if (zmqContext != nullptr)
    {
        zmq_ctx_term (zmqContext);
        zmqContext = nullptr;
    }

    boundZmqPort = 0;
}

void SpikeSpyStreamer::publishSweep (const SlicedSweep& sweep, const float* waveform, int numSamples)
{
    if (zmqPublisher == nullptr || waveform == nullptr || numSamples <= 0)
        return;

    std::ostringstream jsonStream;
    jsonStream << "{"
               << "\"type\":\"erp_sweep\","
               << "\"sample_rate\":" << sweep.sampleRate << ",";

    jsonStream << std::fixed << std::setprecision (6)
               << "\"stim_timestamp_s\":"
               << (double (sweep.stimSample) / double (sweep.sampleRate)) << ","
               << "\"pre_offset_s\":"
               << (double (sweep.preSamples) / double (sweep.sampleRate)) << ",";

    jsonStream.unsetf (std::ios_base::floatfield);

    jsonStream << "\"num_samples\":" << numSamples << ","
               << "\"rec_channel\":" << sweep.recChannel << ","
               << "\"spike_latencies_s\":{";

    bool firstEntry = true;
    for (const auto& entry : sweep.spikeLatencies)
    {
        if (! firstEntry)
            jsonStream << ",";

        jsonStream << "\"" << entry.first << "\":" << entry.second;
        firstEntry = false;
    }

    jsonStream << "}}";

    const std::string header = jsonStream.str();

    zmq_send (zmqPublisher, header.data(), header.size(), ZMQ_SNDMORE);
    zmq_send (zmqPublisher, waveform, numSamples * sizeof (float), 0);
}

void SpikeSpyStreamer::resolveSweeps (uint16 streamId)
{
    auto ringIter = rings.find (streamId);
    if (ringIter == rings.end())
        return;

    Ring& ring = ringIter->second;
    if (ring.samples.empty() || ring.newestSample < 0)
        return;

    const int capacity = int (ring.samples.size());
    const int64 oldestAvailable = ring.newestSample - (int64 (ring.filled) - 1);

    auto sweepIter = pendingSweeps.begin();
    while (sweepIter != pendingSweeps.end())
    {
        if (sweepIter->streamId != streamId)
        {
            ++sweepIter;
            continue;
        }

        const int totalSamples = sweepIter->preSamples + sweepIter->sweepLengthSamples;
        const int64 startSample = sweepIter->stimSample - sweepIter->preSamples;
        const int64 endSampleInclusive = startSample + int64 (totalSamples) - 1;

        if (endSampleInclusive > ring.newestSample)
        {
            ++sweepIter; // wait until the rest of the window has streamed in
            continue;
        }

        if (startSample < oldestAvailable)
        {
            DBG ("SpikeSpyStreamer: dropping sweep that predates the ring buffer");
            sweepIter = pendingSweeps.erase (sweepIter);
            continue;
        }

        std::vector<float> waveform (totalSamples);

        for (int s = 0; s < totalSamples; ++s)
        {
            const int64 samplesBack = ring.newestSample - (startSample + s);
            int slot = ring.writePos - 1 - int (samplesBack);
            slot %= capacity;
            if (slot < 0)
                slot += capacity;

            waveform[s] = ring.samples[slot];
        }

        publishSweep (*sweepIter, waveform.data(), totalSamples);
        sweepIter = pendingSweeps.erase (sweepIter);
    }
}

void SpikeSpyStreamer::process (AudioBuffer<float>& continuousBuffer)
{
    checkForEvents (true);

    std::lock_guard<std::mutex> lock (stateMutex);

    for (auto stream : getDataStreams())
    {
        if (! (*stream)["enable_stream"])
            continue;

        const uint16 streamId = stream->getStreamId();
        const uint32 numSamples = getNumSamplesInBlock (streamId);
        if (numSamples == 0)
            continue;

        const float sampleRate = stream->getSampleRate();
        const int capacity = std::max (1, int (sampleRate * float (RING_SECONDS)) + 1);

        Ring& ring = rings[streamId];
        if (int (ring.samples.size()) != capacity)
        {
            ring.samples.assign (capacity, 0.0f);
            ring.writePos = 0;
            ring.filled = 0;
        }

        const int recChannel = cachedRecChannel;
        const int localChannel = jlimit (0, stream->getChannelCount() - 1, recChannel);
        const int globalChannel = getGlobalChannelIndex (streamId, localChannel);
        if (globalChannel < 0)
            continue;

        const float* source = continuousBuffer.getReadPointer (globalChannel);

        for (uint32 i = 0; i < numSamples; ++i)
        {
            ring.samples[ring.writePos] = source[i];
            ring.writePos = (ring.writePos + 1) % capacity;
        }

        ring.filled = int (std::min<int64_t> (int64_t (ring.filled) + int64_t (numSamples), int64_t (capacity)));
        ring.newestSample = getFirstSampleNumberForBlock (streamId) + int64_t (numSamples) - 1;

        resolveSweeps (streamId);
    }
}

void SpikeSpyStreamer::handleTTLEvent (TTLEventPtr event)
{
    if (event == nullptr || ! event->getState())
        return;

    if (event->getLine() != uint8 (jlimit (1, 16, cachedTtlLine) - 1))
        return;

    DataStream* stream = getDataStream (event->getStreamId());
    if (stream == nullptr)
        return;

    SlicedSweep sweep;
    sweep.streamId = event->getStreamId();
    sweep.stimSample = event->getSampleNumber();
    sweep.sampleRate = stream->getSampleRate();
    sweep.preSamples = std::max (0, int (cachedPreStimMs / 1000.0f * sweep.sampleRate));
    sweep.sweepLengthSamples = std::max (1, int (cachedWindowMs / 1000.0f * sweep.sampleRate));
    sweep.recChannel = cachedRecChannel;

    std::lock_guard<std::mutex> lock (stateMutex);
    pendingSweeps.push_back (sweep);

    if (int (pendingSweeps.size()) > MAX_PENDING_SWEEPS)
        pendingSweeps.erase (pendingSweeps.begin());
}

void SpikeSpyStreamer::handleSpike (SpikePtr spike)
{
    if (spike == nullptr)
        return;

    std::lock_guard<std::mutex> lock (stateMutex);
    if (pendingSweeps.empty())
        return;

    const uint16 streamId = spike->getStreamId();
    const int64 sampleNumber = spike->getSampleNumber();
    const std::string unitName = "unit_" + std::to_string (spike->getSortedId());

    for (auto& sweep : pendingSweeps)
    {
        if (sweep.streamId != streamId)
            continue;

        const int64 startSample = sweep.stimSample - sweep.preSamples;
        const int64 endSampleInclusive = startSample + int64_t (sweep.preSamples) + int64_t (sweep.sweepLengthSamples) - 1;

        if (sampleNumber < startSample || sampleNumber > endSampleInclusive)
            continue;

        const float latencyS = float (int64_t (sampleNumber) - int64_t (sweep.stimSample)) / sweep.sampleRate;
        sweep.spikeLatencies[unitName] = latencyS;
    }
}

void SpikeSpyStreamer::parameterValueChanged (Parameter* param)
{
    if (param == nullptr)
        return;

    const String paramName = param->getName();

    std::lock_guard<std::mutex> lock (stateMutex);

    if (paramName.equalsIgnoreCase ("ttl_line"))
    {
        cachedTtlLine = (int) param->getValue();
    }
    else if (paramName.equalsIgnoreCase ("rec_channel"))
    {
        cachedRecChannel = (int) param->getValue();
    }
    else if (paramName.equalsIgnoreCase ("window_ms"))
    {
        cachedWindowMs = (float) param->getValue();
    }
    else if (paramName.equalsIgnoreCase ("pre_stim_ms"))
    {
        cachedPreStimMs = (float) param->getValue();
    }
    else if (paramName.equalsIgnoreCase ("zmq_port"))
    {
        cachedZmqPort = (int) param->getValue();

        if (! acquiring && cachedZmqPort != boundZmqPort)
            bindZmq (cachedZmqPort);
    }
}

void SpikeSpyStreamer::updateSettings()
{
    std::lock_guard<std::mutex> lock (stateMutex);
    rings.clear();
    pendingSweeps.clear();
}

bool SpikeSpyStreamer::startAcquisition()
{
    std::lock_guard<std::mutex> lock (stateMutex);
    acquiring = true;
    rings.clear();
    pendingSweeps.clear();

    refreshParameterCache();

    if (cachedZmqPort != boundZmqPort)
        bindZmq (cachedZmqPort);

    return true;
}

bool SpikeSpyStreamer::stopAcquisition()
{
    std::lock_guard<std::mutex> lock (stateMutex);
    acquiring = false;
    rings.clear();
    pendingSweeps.clear();

    return true;
}

String SpikeSpyStreamer::handleConfigMessage (const String& msg)
{
    DBG ("SpikeSpyStreamer: config message (ignored): " << msg);
    return String();
}

void SpikeSpyStreamer::handleBroadcastMessage (const String& msg, const int64 systemTimeMillis)
{
    DBG ("SpikeSpyStreamer: broadcast message (ignored): " << msg);
}
