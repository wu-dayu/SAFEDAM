# Checkpoints

Place model checkpoint files in this directory before running the trackers.

`sam3.pt` is required by the current tracking wrappers and must be obtained manually. The wrappers look for it at:

```text
checkpoints/sam3.pt
```

Do not commit downloaded checkpoint binaries unless the project explicitly decides to track them.
