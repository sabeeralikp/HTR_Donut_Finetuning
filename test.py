import re
import transformers
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import random
import numpy as np
from datasets import load_from_disk

# hidde logs
transformers.logging.disable_default_handler()


# Load our model from Hugging Face
processor = DonutProcessor.from_pretrained("donut-base-iam-line")
model = VisionEncoderDecoderModel.from_pretrained("donut-base-iam-line/checkpoint-300")

# Move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# processed_dataset = load_from_disk("processed_dataset")
# Load random document image from the test set
# test_sample = processed_dataset["test"][random.randint(1, 50)]
test_sample = {}
test_sample["image"] = Image.open("test_image/test2.jpg").convert("RGB")
test_sample["target_sequence"] = "In the name of most gracious"
# test_sample["pixel_values"] = processor(
#     test_sample["image"], return_tensors="pt"
# ).pixel_values


def run_prediction(sample, model=model, processor=processor):
    # prepare inputs
    # pixel_values = torch.tensor(test_sample["pixel_values"]).unsqueeze(0)
    pixel_values = processor(test_sample["image"], return_tensors="pt").pixel_values
    task_prompt = "<s>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids

    # run inference
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # process output
    prediction = processor.batch_decode(outputs.sequences)[0]
    prediction = processor.token2json(prediction)

    # load reference target
    target = processor.token2json(test_sample["target_sequence"])
    return prediction, target


prediction, target = run_prediction(test_sample)
print(f"Reference:\n {target}")
print(f"Prediction:\n {prediction}")
# processor.feature_extractor.to_pil_image(np.array(test_sample["pixel_values"])).resize(
#     (350, 600)
# )


# from tqdm import tqdm

# # define counter for samples
# true_counter = 0
# total_counter = 0
# # iterate over dataset
# for sample in tqdm(processed_dataset["test"]):
#     prediction, target = run_prediction(test_sample)
#     for s in zip(prediction.values(), target.values()):
#         if s[0] == s[1]:
#             true_counter += 1
#         total_counter += 1

# print(f"Accuracy: {(true_counter/total_counter)*100}%")
# # Accuracy: 75.0%
