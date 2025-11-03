Place your seeded model artifacts here so the container can copy them to the
mounted volume on first boot.

Expected layout (relative to this folder):

  digits/<MODEL_ID>/model.pt
  digits/<MODEL_ID>/manifest.json

Example:

  digits/mnist_resnet18_v1/model.pt
  digits/mnist_resnet18_v1/manifest.json

At runtime, scripts/prestart.py copies files from /seed/digits/<MODEL_ID>/ into
/data/digits/<MODEL_ID>/ only if the destination files are missing. This allows
you to attach a persistent volume at /data and keep your models across
restarts; the seed is used only the first time.

