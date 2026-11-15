
- refactor out calculations to some processing step not part of the views themselves (code readability)
- ensure the new PGViews keep all the same features as the original
- fix the parameters window to only show parametrs of latest valid window (not just jumping on every window)
    - have the windows register their options to this paramters window rather than the bar above
- unit footprint should have option to show the live latest trace aligned to the most recent stim if it doesnt hit or does?

- the single channel view with multiple channels needs mroe testing 
- the single channel view should nto downsample.

- move the state? should the channelgroup hold effectively the viewerstate? the windows then work from there?
- have consistent live-options (e.g. stimulation following, or offset from selected stimno). 
- keyboard shortcuts to stop following, switch to edit/read mode for a visualiser

- option for demo to play back existing h5 would be cool - but maybe i should just try from oe

- the global UI thread still hangs quite a lot - is that a python limitation or a QT rendering all the images issue.