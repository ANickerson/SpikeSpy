/*
    ------------------------------------------------------------------

    SpikeSpyStreamerEditor - Open Ephys GUI v1.x Processor Editor (API 10)

    Displays processor configuration controls and live bidirectional
    connection / heartbeat status between Open Ephys and SpikeSpy.

    ------------------------------------------------------------------
*/

#include "SpikeSpyStreamerEditor.h"
#include "SpikeSpyStreamer.h"

SpikeSpyStreamerEditor::SpikeSpyStreamerEditor (GenericProcessor* parentNode)
    : GenericEditor (parentNode),
      streamer (static_cast<SpikeSpyStreamer*> (parentNode))
{
    desiredWidth = 280;

    // -------------------------------------------------------------
    // Parameter editors (left and center columns)
    // -------------------------------------------------------------
    addTextBoxParameterEditor (Parameter::PROCESSOR_SCOPE, "ttl_line", 10, 24);
    addTextBoxParameterEditor (Parameter::PROCESSOR_SCOPE, "rec_channel", 10, 58);
    addTextBoxParameterEditor (Parameter::PROCESSOR_SCOPE, "zmq_port", 10, 92);

    addTextBoxParameterEditor (Parameter::PROCESSOR_SCOPE, "window_ms", 85, 24);
    addTextBoxParameterEditor (Parameter::PROCESSOR_SCOPE, "pre_stim_ms", 85, 58);

    for (auto ed : parameterEditors)
    {
        ed->setLayout (ParameterEditor::Layout::nameOnTop);
        ed->setSize (70, 32);
    }

    // -------------------------------------------------------------
    // SpikeSpy Connection & Activity Status Box (right column)
    // -------------------------------------------------------------
    statusTitleLabel = std::make_unique<Label> ("StatusTitle", "SpikeSpy Status");
    statusTitleLabel->setBounds (165, 22, 105, 15);
    statusTitleLabel->setFont (FontOptions ("Inter", "Bold", 12.0f));
    statusTitleLabel->setJustificationType (Justification::centred);
    addAndMakeVisible (statusTitleLabel.get());

    statusBadge = std::make_unique<Label> ("StatusBadge", "WAITING...");
    statusBadge->setBounds (165, 39, 105, 22);
    statusBadge->setFont (FontOptions ("Inter", "Bold", 11.0f));
    statusBadge->setJustificationType (Justification::centred);
    statusBadge->setColour (Label::backgroundColourId, Colour (0xff263238));
    statusBadge->setColour (Label::textColourId, Colour (0xffffb74d));
    statusBadge->setColour (Label::outlineColourId, Colour (0xffff9800));
    addAndMakeVisible (statusBadge.get());

    const int zmqPort = (streamer != nullptr) ? streamer->getZmqPort() : 5556;
    const int hbPort = (streamer != nullptr) ? streamer->getHeartbeatPort() : 5557;

    detailsLabel = std::make_unique<Label> ("DetailsLabel", "Port: " + String (zmqPort) + " / " + String (hbPort));
    detailsLabel->setBounds (165, 63, 105, 14);
    detailsLabel->setFont (FontOptions ("Inter", "Regular", 10.0f));
    detailsLabel->setJustificationType (Justification::centred);
    detailsLabel->setColour (Label::textColourId, Colours::grey);
    addAndMakeVisible (detailsLabel.get());

    sweepsLabel = std::make_unique<Label> ("SweepsLabel", "Sweeps sent: 0");
    sweepsLabel->setBounds (165, 78, 105, 14);
    sweepsLabel->setFont (FontOptions ("Inter", "Regular", 10.0f));
    sweepsLabel->setJustificationType (Justification::centred);
    sweepsLabel->setColour (Label::textColourId, Colours::lightgrey);
    addAndMakeVisible (sweepsLabel.get());

    activityLabel = std::make_unique<Label> ("ActivityLabel", "No connection");
    activityLabel->setBounds (165, 93, 105, 14);
    activityLabel->setFont (FontOptions ("Inter", "Regular", 9.5f));
    activityLabel->setJustificationType (Justification::centred);
    activityLabel->setColour (Label::textColourId, Colours::grey);
    addAndMakeVisible (activityLabel.get());

    startTimerHz (3);
}

SpikeSpyStreamerEditor::~SpikeSpyStreamerEditor()
{
    stopTimer();
}

void SpikeSpyStreamerEditor::timerCallback()
{
    if (streamer == nullptr)
        return;

    const bool connected = streamer->isClientConnected();
    const int sweeps = streamer->getSweepsSent();

    if (connected != lastConnectedState)
    {
        lastConnectedState = connected;

        if (connected)
        {
            statusBadge->setText ("CONNECTED", dontSendNotification);
            statusBadge->setColour (Label::backgroundColourId, Colour (0xff1b5e20));
            statusBadge->setColour (Label::textColourId, Colour (0xffa5d6a7));
            statusBadge->setColour (Label::outlineColourId, Colour (0xff4caf50));

            detailsLabel->setText ("Client: " + streamer->getClientName(), dontSendNotification);
            detailsLabel->setColour (Label::textColourId, Colours::white);

            activityLabel->setText ("Heartbeat: Active", dontSendNotification);
            activityLabel->setColour (Label::textColourId, Colour (0xff81c784));
        }
        else
        {
            statusBadge->setText ("WAITING...", dontSendNotification);
            statusBadge->setColour (Label::backgroundColourId, Colour (0xff263238));
            statusBadge->setColour (Label::textColourId, Colour (0xffffb74d));
            statusBadge->setColour (Label::outlineColourId, Colour (0xffff9800));

            detailsLabel->setText ("Port: " + String (streamer->getZmqPort()) + " / " + String (streamer->getHeartbeatPort()), dontSendNotification);
            detailsLabel->setColour (Label::textColourId, Colours::grey);

            activityLabel->setText ("No connection", dontSendNotification);
            activityLabel->setColour (Label::textColourId, Colours::grey);
        }
    }

    if (sweeps != lastSweepsCount)
    {
        lastSweepsCount = sweeps;
        sweepsLabel->setText ("Sweeps sent: " + String (sweeps), dontSendNotification);
    }
}
