import json
import logging
import math
import os
import random
import shutil

import accelerate as acc
import datasets
import diffusers
import numpy as np
import pandas as pd
import peft
import PIL
import torch
import torchvision
import transformers
from tqdm.auto import tqdm

from .lora_hyperparameters import HYPERPARAMETERS

CWD = os.getcwd()
logger = logging.getLogger(__name__)


#
# Functions for dataset handling
#
def _create_dataset_with_captions(
    dataset_name: str = "test_dataset",
) -> datasets.Dataset:
    features = datasets.Features({
        "image": datasets.Image(),
        "caption": datasets.Value(dtype="string"),
    })

    data_dir = os.path.join(CWD, "data", "datasets", "train")

    with open(os.path.join(data_dir, "captions.json")) as file:
        captions = json.load(file)
    images = [
        datasets.Image().encode_example(PIL.Image.open(os.path.join(data_dir, file)))
        for file in captions.keys()
    ]

    df = pd.DataFrame({"caption": list(captions.values()), "image": images})

    dataset = datasets.Dataset.from_pandas(df, features=features)
    # dataset.save_to_disk(dataset_path=os.path.join(data_dir,'blah'))
    return {"train": dataset}


def _load_dataset(
    dataset_name: str = "test_dataset", separate_captions_file: str = None
):
    dataset = datasets.load_dataset(
        "imagefolder", data_dir=os.path.join(CWD, "data", "datasets")
    )

    if separate_captions_file is not None:
        with open(
            os.path.join(CWD, "data", "datasets", "captions.json")
        ) as file:
            dataset.add_column("caption", json.load(file).values())

    return dataset


#
# Functions for model configuration handling
#
def _import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder"
):
    text_encoder_config = transformers.PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


#
# Functions for tokenizing prompts
#
def _tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    """Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt"""
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = _tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
            return_dict=False,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


#
# Main module function
#
def train_lora(args):
    # 1. Configure stuff
    logging_dir = os.path.join(CWD, "logs", "training", args.dir)
    output_dir = os.path.join(CWD, "models", "training", args.dir)
    pretrained_model_name_or_path = os.path.join(
        CWD, "models", HYPERPARAMETERS.get("checkpoint")
    )

    gradient_accumulation_steps = HYPERPARAMETERS.get("gradient_accumulation_steps")
    seed = HYPERPARAMETERS.get("seed")

    accelerator_project_config = acc.utils.ProjectConfiguration(
        project_dir=output_dir, logging_dir=logging_dir
    )
    kwargs = acc.utils.DistributedDataParallelKwargs(
        find_unused_parameters=True
    )  # TODO remove?
    accelerator = acc.Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        # mixed_precision=HYPERPARAMETERS.mixed_precision, # TODO remove?
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],  # TODO remove?
    )

    # Make one log on every process with the configuration for debugging.
    logger.info(accelerator.state)
    # if accelerator.is_local_main_process:
    #     datasets.utils.logging.set_verbosity_warning() # TODO remove?
    #     transformers.utils.logging.set_verbosity_warning() # TODO remove?
    #     diffusers.utils.logging.set_verbosity_info() # TODO remove?
    # else:
    #     datasets.utils.logging.set_verbosity_error() # TODO remove?
    #     transformers.utils.logging.set_verbosity_error() # TODO remove?
    #     diffusers.utils.logging.set_verbosity_error() # TODO remove?

    if seed is not None:
        acc.utils.set_seed(seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        if logging_dir is not None:
            os.makedirs(logging_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer_one = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    tokenizer_two = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        use_fast=False,
    )

    # import correct text encoder classes # TODO simplify for stable diffusion?
    text_encoder_cls_one = _import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path
    )
    text_encoder_cls_two = _import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = diffusers.DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_2"
    )
    vae_path = (
        pretrained_model_name_or_path
        # if pretrained_vae_model_name_or_path is None # TODO remove?
        # else pretrained_vae_model_name_or_path # TODO remove?
    )
    vae = diffusers.AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae",  # TODO correct for stable diffusion?
        # subfolder="vae" if pretrained_vae_model_name_or_path is None else None # TODO remove?
    )
    unet = diffusers.UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    unet.to(accelerator.device, dtype=weight_dtype)

    # TODO remove?
    # if pretrained_vae_model_name_or_path is None:
    #     vae.to(accelerator.device, dtype=torch.float32)
    # else:
    #    vae.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if HYPERPARAMETERS.get('enable_npu_flash_attention'):
        if diffusers.utils.import_utils.is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            unet.enable_npu_flash_attention()
        else:
            raise ValueError(
                "npu flash attention requires torch_npu extensions and is supported"
                " only on npu devices."
            )

    # now we will add new LoRA weights to the attention layers
    # Set correct lora layers
    unet_lora_config = peft.LoraConfig(
        r=HYPERPARAMETERS.get('rank'),
        lora_alpha=HYPERPARAMETERS.get('rank'),
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.add_adapter(unet_lora_config)

    # The text encoder comes from 🤗 transformers, we will also attach adapters to it.
    if HYPERPARAMETERS.get('train_text_encoder'):
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_lora_config = peft.LoraConfig(
            r=HYPERPARAMETERS.get('rank'),
            lora_alpha=HYPERPARAMETERS.get('rank'),
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)

    # TODO move outside function scope
    def _unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = (
            model._orig_mod
            if diffusers.utils.torch_utils.is_compiled_module(model)
            else model
        )
        return model

    # TODO move outside function scope
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def _save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return

        # there are only two options here. Either are just the unet attn processor layers
        # or there are the unet and text encoder attn layers
        unet_lora_layers_to_save = None
        text_encoder_one_lora_layers_to_save = None
        text_encoder_two_lora_layers_to_save = None

        for model in models:
            if isinstance(_unwrap_model(model), type(_unwrap_model(unet))):
                unet_lora_layers_to_save = (
                    diffusers.utils.convert_state_dict_to_diffusers(
                        peft.utils.get_peft_model_state_dict(model)
                    )
                )
            elif isinstance(
                _unwrap_model(model), type(_unwrap_model(text_encoder_one))
            ):
                text_encoder_one_lora_layers_to_save = (
                    diffusers.utils.convert_state_dict_to_diffusers(
                        peft.utils.get_peft_model_state_dict(model)
                    )
                )
            elif isinstance(
                _unwrap_model(model), type(_unwrap_model(text_encoder_two))
            ):
                text_encoder_two_lora_layers_to_save = (
                    diffusers.utils.convert_state_dict_to_diffusers(
                        peft.utils.get_peft_model_state_dict(model)
                    )
                )
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            if weights:
                weights.pop()

        # TODO question: why this specific pipeline?
        diffusers.StableDiffusionXLPipeline.save_lora_weights(
            output_dir,
            unet_lora_layers=unet_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
        )

    # TODO move outside function scope
    def _load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(_unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(_unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(_unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = diffusers.loaders.LoraLoaderMixin.lora_state_dict(
            input_dir
        )
        unet_state_dict = {
            f'{k.replace("unet.", "")}': v
            for k, v in lora_state_dict.items()
            if k.startswith("unet.")
        }
        unet_state_dict = diffusers.utils.convert_unet_state_dict_to_peft(
            unet_state_dict
        )
        incompatible_keys = peft.set_peft_model_state_dict(
            unet_, unet_state_dict, adapter_name="default"
        )
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    "Loading adapter weights from state_dict led to unexpected keys"
                    f" not found in the model:  {unexpected_keys}. "
                )

        if HYPERPARAMETERS.get('train_text_encoder'):
            diffusers.training_utils._set_state_dict_into_text_encoder(
                lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_
            )

            diffusers.training_utils._set_state_dict_into_text_encoder(
                lora_state_dict,
                prefix="text_encoder_2.",
                text_encoder=text_encoder_two_,
            )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if HYPERPARAMETERS.get('mixed_precision') == "fp16":
            models = [unet_]
            if HYPERPARAMETERS.get('train_text_encoder'):
                models.extend([text_encoder_one_, text_encoder_two_])
            diffusers.training_utils.cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(_save_model_hook)
    accelerator.register_load_state_pre_hook(_load_model_hook)

    if HYPERPARAMETERS.get('gradient_checkpointing'):
        unet.enable_gradient_checkpointing()
        if HYPERPARAMETERS.get('train_text_encoder'):
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    # Make sure the trainable params are in float32.
    if HYPERPARAMETERS.get('mixed_precision') == "fp16":
        models = [unet]
        if HYPERPARAMETERS.get('train_text_encoder'):
            models.extend([text_encoder_one, text_encoder_two])
        diffusers.training_utils.cast_training_params(models, dtype=torch.float32)

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if HYPERPARAMETERS.get('use_8bit_adam'):
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip"
                " install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if HYPERPARAMETERS.get('train_text_encoder'):
        params_to_optimize = (
            params_to_optimize
            + list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
            + list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))
        )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=HYPERPARAMETERS.get('learning_rate'),
        betas=(HYPERPARAMETERS.get('adam_beta1'), HYPERPARAMETERS.get('adam_beta2')),
        weight_decay=HYPERPARAMETERS.get('adam_weight_decay'),
        eps=HYPERPARAMETERS.get('adam_epsilon'),
    )

    # Preprocessing the datasets.
    # Load training dataset
    dataset = _create_dataset_with_captions()
    # dataset = _load_dataset()

    # We need to tokenize inputs and targets.
    image_column, caption_column = dataset["train"].column_names

    # TODO extract
    # We need to tokenize input captions and transform the images.
    def _tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings"
                    " or lists of strings."
                )
        tokens_one = _tokenize_prompt(tokenizer_one, captions)
        tokens_two = _tokenize_prompt(tokenizer_two, captions)
        return tokens_one, tokens_two

    # TODO wtf, why here? it is used in _preprocess_train
    resolution = HYPERPARAMETERS.get('resolution')
    center_crop = HYPERPARAMETERS.get('center_crop')
    random_flip = HYPERPARAMETERS.get('random_flip')

    train_resize = torchvision.transforms.Resize(
        resolution,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )
    train_crop = (
        torchvision.transforms.CenterCrop(resolution)
        if center_crop
        else torchvision.transforms.RandomCrop(resolution)
    )
    train_flip = torchvision.transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5]),
    ])

    # TODO extract
    def _preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        # image augmentation
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if center_crop:
                y1 = max(0, int(round((image.height - resolution) / 2.0)))
                x1 = max(0, int(round((image.width - resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (resolution, resolution))
                image = torchvision.transforms.functional.crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        tokens_one, tokens_two = _tokenize_captions(examples)
        examples["input_ids_one"] = tokens_one
        examples["input_ids_two"] = tokens_two
        if HYPERPARAMETERS.get('debug_loss'):
            fnames = [
                os.path.basename(image.filename)
                for image in examples[image_column]
                if image.filename
            ]
            if fnames:
                examples["filenames"] = fnames
        return examples

    with accelerator.main_process_first():
        if HYPERPARAMETERS.get('max_train_samples') is not None:
            dataset["train"] = (
                dataset["train"]
                .shuffle(seed=HYPERPARAMETERS.get('seed'))
                .select(range(HYPERPARAMETERS.get('max_train_samples')))
            )
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(
            _preprocess_train, output_all_columns=True
        )

    # TODO extract
    def _collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        original_sizes = [example["original_sizes"] for example in examples]
        crop_top_lefts = [example["crop_top_lefts"] for example in examples]
        input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
        input_ids_two = torch.stack([example["input_ids_two"] for example in examples])
        result = {
            "pixel_values": pixel_values,
            "input_ids_one": input_ids_one,
            "input_ids_two": input_ids_two,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
        }

        filenames = [
            example["filenames"] for example in examples if "filenames" in example
        ]
        if filenames:
            result["filenames"] = filenames
        return result

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=_collate_fn,
        batch_size=HYPERPARAMETERS.get('train_batch_size'),
        num_workers=HYPERPARAMETERS.get('dataloader_num_workers'),
    )

    # Scheduler and math around the number of training steps.
    gradient_accumulation_steps = HYPERPARAMETERS.get('gradient_accumulation_steps')
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    max_train_steps = HYPERPARAMETERS.get('num_train_epochs') * num_update_steps_per_epoch

    lr_scheduler = diffusers.optimization.get_scheduler(
        HYPERPARAMETERS.get('lr_scheduler'),
        optimizer=optimizer,
        num_warmup_steps=HYPERPARAMETERS.get('lr_warmup_steps')
        * gradient_accumulation_steps,
        num_training_steps=max_train_steps
        * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    if HYPERPARAMETERS.get('train_text_encoder'):
        (
            unet,
            text_encoder_one,
            text_encoder_two,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet,
            text_encoder_one,
            text_encoder_two,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    max_train_steps = HYPERPARAMETERS.get('num_train_epochs') * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    # TODO clean up this mess with the recalculations
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = (
        HYPERPARAMETERS.get('train_batch_size')
        * accelerator.num_processes
        * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {HYPERPARAMETERS.get('train_batch_size')}"
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) ="
        f" {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    resume_from_checkpoint = HYPERPARAMETERS.get('resume_from_checkpoint')
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{resume_from_checkpoint}' does not exist."
                " Starting a new training run."
            )
            HYPERPARAMETERS.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for _ in range(first_epoch, num_train_epochs):
        unet.train()
        if HYPERPARAMETERS.get('train_text_encoder'):
            text_encoder_one.train()
            text_encoder_two.train()
        train_loss = 0.0
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                # if args.pretrained_vae_model_name_or_path is not None: # TODO remove?
                #     pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                # else:
                #     pixel_values = batch["pixel_values"]
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
                # if args.pretrained_vae_model_name_or_path is None:
                #     model_input = model_input.to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                if HYPERPARAMETERS.get('noise_offset'):
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += HYPERPARAMETERS.get('noise_offset') * torch.randn(
                        (model_input.shape[0], model_input.shape[1], 1, 1),
                        device=model_input.device,
                    )

                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=model_input.device,
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(
                    model_input, noise, timesteps
                )

                # time ids
                def _compute_time_ids(original_size, crops_coords_top_left):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (
                        resolution,
                        resolution,
                    )
                    add_time_ids = list(
                        original_size + crops_coords_top_left + target_size
                    )
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(
                        accelerator.device, dtype=weight_dtype
                    )
                    return add_time_ids

                add_time_ids = torch.cat([
                    _compute_time_ids(s, c)
                    for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
                ])

                # Predict the noise residual
                unet_added_conditions = {"time_ids": add_time_ids}
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders=[text_encoder_one, text_encoder_two],
                    tokenizers=None,
                    prompt=None,
                    text_input_ids_list=[
                        batch["input_ids_one"],
                        batch["input_ids_two"],
                    ],
                )
                unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if HYPERPARAMETERS.get('prediction_type') is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(
                        prediction_type=HYPERPARAMETERS.get('prediction_type')
                    )

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(
                        "Unknown prediction type"
                        f" {noise_scheduler.config.prediction_type}"
                    )

                if HYPERPARAMETERS.get('snr_gamma') is None:
                    loss = torch.nn.functional.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = diffusers.training_utils.compute_snr(
                        noise_scheduler, timesteps
                    )
                    mse_loss_weights = torch.stack(
                        [snr, HYPERPARAMETERS.get('snr_gamma') * torch.ones_like(timesteps)],
                        dim=1,
                    ).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = diffusers.training_utils.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()
                if HYPERPARAMETERS.get('debug_loss') and "filenames" in batch:
                    for fname in batch["filenames"]:
                        accelerator.log({"loss_for_" + fname: loss}, step=global_step)
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(HYPERPARAMETERS.get('train_batch_size'))
                ).mean()
                train_loss += (
                    avg_loss.item() / gradient_accumulation_steps
                )

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        params_to_optimize, HYPERPARAMETERS.get('max_grad_norm')
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if (
                    accelerator.distributed_type == acc.utils.DistributedType.DEEPSPEED
                    or accelerator.is_main_process
                ):
                    if global_step % HYPERPARAMETERS.get('checkpointing_steps') == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if HYPERPARAMETERS.get('checkpoints_total_limit') is not None:
                            checkpoints = os.listdir(output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if (
                                len(checkpoints)
                                >= HYPERPARAMETERS.get('checkpoints_total_limit')
                            ):
                                num_to_remove = (
                                    len(checkpoints)
                                    - HYPERPARAMETERS.get('checkpoints_total_limit')
                                    + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist,"
                                    f" removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    "removing checkpoints:"
                                    f" {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = _unwrap_model(unet)
        unet_lora_state_dict = diffusers.utils.convert_state_dict_to_diffusers(
            peft.utils.get_peft_model_state_dict(unet)
        )

        if HYPERPARAMETERS.get('train_text_encoder'):
            text_encoder_one = _unwrap_model(text_encoder_one)
            text_encoder_two = _unwrap_model(text_encoder_two)

            text_encoder_lora_layers = diffusers.utils.convert_state_dict_to_diffusers(
                peft.utils.get_peft_model_state_dict(text_encoder_one)
            )
            text_encoder_2_lora_layers = (
                diffusers.utils.convert_state_dict_to_diffusers(
                    peft.utils.get_peft_model_state_dict(text_encoder_two)
                )
            )
        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None

        diffusers.StableDiffusionXLPipeline.save_lora_weights(
            save_directory=output_dir,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )

        del unet
        del text_encoder_one
        del text_encoder_two
        del text_encoder_lora_layers
        del text_encoder_2_lora_layers
        torch.cuda.empty_cache()

        # Final inference
        # Make sure vae.dtype is consistent with the unet.dtype
        if HYPERPARAMETERS.get('mixed_precision') == "fp16":
            vae.to(weight_dtype)
        # Load previous pipeline
        pipeline = diffusers.StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path,
            vae=vae,
            torch_dtype=weight_dtype,
        )
        pipeline = pipeline.to(accelerator.device)

        # load attention processors
        pipeline.load_lora_weights(output_dir)

    accelerator.end_training()
