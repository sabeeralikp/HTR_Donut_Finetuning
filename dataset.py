from datasets import load_dataset, load_from_disk
from transformers import DonutProcessor
import multiprocessing


def json2token(
    obj, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True
):
    """
    Convert an ordered JSON object into a token sequence
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                if update_special_tokens_for_json_key:
                    (
                        new_special_tokens.append(rf"<s_{k}>")
                        if rf"<s_{k}>" not in new_special_tokens
                        else None
                    )
                    (
                        new_special_tokens.append(rf"</s_{k}>")
                        if rf"</s_{k}>" not in new_special_tokens
                        else None
                    )
                output += (
                    rf"<s_{k}>"
                    + json2token(
                        obj[k], update_special_tokens_for_json_key, sort_json_key
                    )
                    + rf"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join(
            [
                json2token(item, update_special_tokens_for_json_key, sort_json_key)
                for item in obj
            ]
        )
    else:
        # excluded special tokens for now
        obj = str(obj)
        if f"<{obj}/>" in new_special_tokens:
            obj = f"<{obj}/>"  # for categorical special tokens
        return obj


def preprocess_documents_for_donut(sample):
    # create Donut-style input
    # print(sample['text'])
    # text = json.loads(sample["text"])
    d_doc = task_start_token + json2token(sample["text"]) + eos_token
    # convert all images to RGB
    image = sample["image"].convert("RGB")
    return {"image": image, "text": d_doc}


# Load processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")


def transform_and_tokenize(
    sample, processor=processor, split="train", max_length=512, ignore_id=-100
):
    # create tensor from image
    try:
        pixel_values = processor(
            sample["image"], random_padding=split == "train", return_tensors="pt"
        ).pixel_values.squeeze()
    except Exception as e:
        print(sample)
        print(f"Error: {e}")
        return {}

    # tokenize document
    input_ids = processor.tokenizer(
        sample["text"],
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].squeeze(0)

    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = (
        ignore_id  # model doesn't need to predict pad token
    )
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "target_sequence": sample["text"],
    }


dataset = load_dataset("Teklia/IAM-line")
dataset["train"] = dataset["train"].select(range(999))
dataset["validation"] = dataset["validation"].select(range(499))
dataset["test"] = dataset["test"].select(range(499))

print(dataset["train"])
print(f"Dataset features are: {dataset.keys()}")


new_special_tokens = []  # new tokens which will be added to the tokenizer
task_start_token = "<s>"  # start of task token
eos_token = "</s>"  # eos token of tokenizer


proc_dataset = dataset.map(
    preprocess_documents_for_donut, num_proc=multiprocessing.cpu_count()
)

# add new special tokens to tokenizer
processor.tokenizer.add_special_tokens(
    {"additional_special_tokens": new_special_tokens + [task_start_token] + [eos_token]}
)

# we update some settings which differ from pretraining; namely the size of the images + no rotation required
# resizing the image to smaller sizes from [1920, 2560] to [960,1280]
processor.feature_extractor.size = [640, 854]  # should be (width, height)
processor.feature_extractor.do_align_long_axis = False


#

processed_dataset = proc_dataset.map(
    transform_and_tokenize,
    remove_columns=["image", "text"],
)

# # COMMENT IN in case you want to save the processed dataset to disk in case of error later
print("Saving Pre-Processed Dataset")
processed_dataset.save_to_disk("processed_dataset")
print("Saving Processor")
processor.save_pretrained("processor")
