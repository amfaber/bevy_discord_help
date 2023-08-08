# bevy_discord_help

A small repo to make questions clearer on the bevy discord

I am trying to morph the shader instancing example into my own custom rendering pipeline that will eventually be used for volume rendering.

There are currently two roadblocks:

1. About 40% of the time when starting the app, its as if the custom rendering pipeline never gets registered. `queue_custom` never runs and nothing from
the custom pipeline is rendered. Restarting the app gives it another chance to actually run the pipeline. I have no idea what could be causing this behaviour

2. When the pipeline does run, the debugging cube now flickers. This behaviour started after I
added the `prepare_render_textures` and added it to the prepare phase of rendering. The function
serves to create screen space textures eventually used by the volume render. I tried following
how bevy's prepass sets up its screen space buffers, but something is off. Perhaps prepare_windows doesn't run on every frame?


