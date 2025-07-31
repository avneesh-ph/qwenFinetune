import random
from datasets import load_dataset
from PIL import Image
from io import BytesIO

class DatasetLoader():
    def __init__(self, path):
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
        

    # Format samples for training
    def format_sample_for_training(self, samples):
        """Convert LVIS sample to Qwen training format"""
        # print(samples)
        prompt = training_prompt()
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "content": self.system_message
                    }
                ]
            },
            {
            "role": "user",
            "content": [
                    {"type": "image"},
                    {"type": "text", "text": "extract data from the image in json"}
                ]
            },
            {
                "role": "assistant", 
                "content": sample["ground_truth"]
            }
        ]
    
        return {
            "messages": conversation,
            "image": sample['image']
        }

    def loadDataset(self):
        # Prepare training data
        print("Preparing training data...")
        train_subset = self._train.select(range(1000))  # Use 1000 samples
        eval_subset = self._validation.select(range(100))  # Use 100 for evaluation

        formatted_train = train_subset.map(self.training_prompt)
        formatted_eval = eval_subset.map(self.format_sample_for_training)

        print(f"Training samples ready: {len(formatted_train)}")
        print(f"Evaluation samples ready: {len(formatted_eval)}")
        return formatted_train, formatted_eval