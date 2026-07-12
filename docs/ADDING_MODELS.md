# Adding a model

Three steps, same pattern as upstream VBVR-InferKit.

## 1. Write (or port) the wrapper

Create `v2vinferkit/models/<provider>_inference.py` with two layers:

- a service class that talks to the provider API (auth from env vars, submit,
  poll, download), and
- a wrapper class subclassing ModelWrapper from `v2vinferkit/models/base.py`
  that implements generate() and returns the 8-field result dict.

For a v2v model, generate() must accept a video_path kwarg and route to the
video-to-video path when it is set. See runway_inference.py for the cleanest
example of the split.

If the model already exists in upstream VBVR-InferKit, just copy the file and
rename the import prefix vbvrinferkit → v2vinferkit.

## 2. Register it

Add an entry to `v2vinferkit/runner/MODEL_CATALOG.py`:

```python
"my-model-v2v": {
    "wrapper_module": "v2vinferkit.models.my_inference",
    "wrapper_class": "MyWrapper",
    "service_class": "MyService",
    "model": "provider-model-id",
    "modality": "v2v",
    "description": "One line about what it does",
    "family": "My Provider"
},
```

Add it to a family dict and to AVAILABLE_MODELS / MODEL_FAMILIES. Also add the
lazy-import entry in `v2vinferkit/models/__init__.py` and the key slot in
env.template.

## 3. Verify

```bash
python3 run.py --list-models                     # entry shows up
python3 run.py --model my-model-v2v --questions-dir examples \
    --output-dir /tmp/out --dry-run              # wiring
python3 run.py --model my-model-v2v --questions-dir examples \
    --output-dir /tmp/out                        # one real call (costs money)
```

Keep new deps out of the core list unless every user needs them.
