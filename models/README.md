# models/

Downloaded ONNX weights live here (they are git-ignored — don't commit them).

## Smart Turn v3.1 (end-of-turn detection)

```bash
python models/download_smart_turn.py
```

Saves `smart-turn-v3.1.onnx`. The media worker reads it via
`SMART_TURN_MODEL_PATH` (default `models/smart-turn-v3.1.onnx`). If it's absent,
the worker runs without end-of-turn detection — every VAD pause is treated as a
complete turn, which is the Phase-1 behaviour.

Under Docker Compose this directory is mounted into the container, so downloading
it once on the host is enough.
