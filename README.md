# bevy_discord_help

A small repo to get some help from the bevy discord

This is still pretty dirty, this is my first attempt at implementing a custom rendering node in bevy and I am still very much learning.
I've noticed that sometimes it seems like "queue_volume_render" never runs, at which point the app runs fine without any panics, but obviously the actual
volume render doesn't occur either. I have no idea what causes this, but closing the app and rerunning seems to bring you to the real crash in most cases.
