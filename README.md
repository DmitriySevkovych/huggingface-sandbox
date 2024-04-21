# HuggingFace Sandbox

## Notes on `transformers`:

### Installation, [cf. official docs](https://huggingface.co/docs/transformers/installation)

#### Cache setup

Pretrained models are downloaded and locally cached at: `~/.cache/huggingface/hub`. This is the default directory given by the shell environment variable `TRANSFORMERS_CACHE`.

####  Offline mode

Run 🤗 Transformers in a firewalled or offline environment with locally cached files by setting the environment variable `TRANSFORMERS_OFFLINE=1`.

Add 🤗 Datasets to your offline training workflow with the environment variable `HF_DATASETS_OFFLINE=1`.

Alternatively, there are three ways to pre-fetch models and tokenizers to use them offline later on.

- Download a file through the user interface on the Model Hub by clicking on the ↓ icon.
- Use the `from_pretrained()` (to download) and `save_pretrained()` (to save locally) methods for model and tokenizer
- Programmatically download files with the `huggingface_hub` library
