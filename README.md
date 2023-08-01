# bevy_discord_help

A small repo to get some help from the bevy discord

This is still pretty dirty, this is my first attempt at implementing a custom rendering node in bevy and I am still very much learning. The workflow
is to try to run the program and fix any panics encountered.

Currently we hit this error, but I can't figure out why there's a discrepency between the render pipelines layout (presumably set by specializing the mesh pipeline) and the bindgroup's (presumably set by SetMeshViewBindGroup<0>).
```
Caused by:
    In a RenderPass
      note: encoder = `<CommandBuffer-(0, 1, Vulkan)>`
    In a draw command, indexed:true indirect:false
      note: render pipeline = `Volume render`
    The pipeline layout, associated with the current render pipeline, contains a bind group layout at index 0 which is incompatible with the bind group layout associated with the bind group at 0
```

I've noticed that sometimes it seems like "queue_volume_render" never runs, at which point the app runs fine without any panics, but obviously the actual
volume render doesn't occur either. I have no idea what causes this, but closing the app and rerunning seems to bring you to the real crash in most cases.
