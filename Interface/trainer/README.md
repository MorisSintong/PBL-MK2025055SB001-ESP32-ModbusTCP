# trainer

Output directory for the trained LBPH face-recognition model.

## Files

- `trainer.yml` – OpenCV `LBPHFaceRecognizer` model produced by
  `02_training.py` (or the `/train_model` endpoint in `app.py`) from the images
  in `../dataset/`. The Flask HMI loads this file at startup to recognize
  operators.

To regenerate the model, delete or replace `trainer.yml` and re-run the
training step. If no users are left in the dataset, the training script
removes `trainer.yml` and the HMI reports "NO MODEL".