# dataset

Storage folder for captured face images used to train the LBPH face recognizer.

## Format

Images are saved by the capture step (either `01_Ambil_Wajah.py` in the parent
`Interface/` directory, or the web HMI registration mode in `app.py`) using the
naming convention:

```
User.<id>.<n>.jpg
```

- `<id>` – numeric user/operator ID (e.g. `1`).
- `<n>`  – sequential capture number (e.g. `1`..`30`).

Example: `User.1.7.jpg` is the 7th grayscale face crop of user ID 1.

The training script (`02_training.py` or `app.py` `/train_model` route) reads
every `User.*.*.jpg` file in this folder, extracts the id from the filename,
and trains `../trainer/trainer.yml`.

This directory is intentionally empty by default – it is populated at runtime.