from datasets import load_dataset, load_from_disk
from pathlib import Path
from PIL import Image
from huggingface_hub import HfFolder
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import os, json, shutil, random, multiprocessing, torch

# # COMMENT IN in case you want to load the processed dataset from disk in case of error later
processed_dataset = load_from_disk("processed_dataset")
processor = DonutProcessor.from_pretrained("processor")

# processed_dataset = processed_dataset.train_test_split(test_size=0.1)
print(processed_dataset)

# Load model from huggingface.co
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Resize embedding layer to match vocabulary size
new_emb = model.decoder.resize_token_embeddings(len(processor.tokenizer))
print(f"New embedding size: {new_emb}")
# Adjust our image size and output sequence lengths
print(processor.feature_extractor.size)
model.config.encoder.image_size = processor.feature_extractor.size[
    "height"
]  # (height, width)
model.config.decoder.max_length = len(
    max(processed_dataset["train"]["labels"], key=len)
)

# Add task token for decoder to start
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
    ["<s>"]
)[0]

# is done by Trainer
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# hyperparameters used for multiple args
hf_repository_id = "donut-base-iam-line"

# Arguments for training
training_args = Seq2SeqTrainingArguments(
    output_dir=hf_repository_id,
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    weight_decay=0.01,
    fp16=True,
    logging_steps=100,
    save_total_limit=2,
    evaluation_strategy="no",
    save_strategy="epoch",
    predict_with_generate=True,
    # push to hub parameters
    report_to="wandb",
    # push_to_hub=True,
    # hub_strategy="every_save",
    # hub_model_id=hf_repository_id,
    # hub_token=HfFolder.get_token(),
)

# Create Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["test"],
)

# Start training
trainer.train()

# Save processor and create model card
processor.save_pretrained(hf_repository_id)
trainer.create_model_card()
# trainer.push_to_hub()
