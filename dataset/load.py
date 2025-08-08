import random
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import os

# Refactored formatting function outside of class
def format_sample_for_training(sample, system_message):

    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "extract data from the image in json"},
                {"type": "image", "image": sample["image"]},
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": sample["ground_truth"]
                }
            ]
        }
    ]
    return {
        "messages": conversation,
    }

# Now your DatasetLoader class:
class DatasetLoader:
    def __init__(self, path, save_images=True):
        self._path = path
        self._dataset = load_dataset(self._path)

        self._train = self._dataset["train"]
        self._validation = self._dataset["validation"]
        self._test = self._dataset["test"]

        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

        self.system_message = """You are a Vision Language Model specialized in parsing forms, which may contain handwritten or system-generated text. Your task is to extract key information from the form and output it in a structured JSON format.

            * The form may include fields such as `NAME`, `DATE`, `CITY`, `STATE`, `ZIP`, etc., but the exact fields can vary.
            * Focus only on extracting clearly labeled and filled fields from the form.
            * Use the field labels present in the image as keys in the JSON output (in uppercase or as they appear).
            * Return the result in the format:

            ```json
            {"gt_parse": {"FIELD_1": "value", "FIELD_2": "value", ...}}
            ```

            * Handwritten text should be transcribed as accurately as possible.
            * Do not include empty or unlabeled fields.
            * Do not provide any explanation—only return the JSON output.

            ---

            **Parsed Output for Uploaded Image:**

            ```json
            {
            "gt_parse": {
                "DATE": "8-3-89",
                "CITY": "MINDEN CITY",
                "STATE": "Mi",
                "ZIP": "48456"
            }
            }
            ```
        """

    def loadDataset(self):
        base_path = os.getcwd()
        
        self.train_subset = self._train.select(range(1000))   # e.g. first 1000 samples for training
        self.eval_subset = self._validation.select(range(100)) # e.g. 100 samples for eval

        formatted_train = [format_sample_for_training(sample, self.system_message) for sample in self.train_subset]
        formatted_eval = [format_sample_for_training(sample, self.system_message) for sample in self.eval_subset]

        # formatted_train = self.train_subset.map(
        #     format_sample_for_training,
        #     fn_kwargs={
        #         "system_message": self.system_message,
        #     },
        # )

        # formatted_eval = self.eval_subset.map(
        #     format_sample_for_training,
        #     fn_kwargs={
        #         "system_message": self.system_message,
        #     },
        # )

        print(f"Training samples ready: {len(formatted_train)}")
        print(f"Evaluation samples ready: {len(formatted_eval)}")
        return formatted_train, formatted_eval