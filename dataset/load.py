import random
from datasets import load_dataset

class DatasetLoader():
    def __init__(self, path):
        self._path = path
        self._dataset = load_dataset(self._path)
        self._train = self._dataset["train"]
        self._validation = self._dataset["validation"]
        self._test = self._dataset["test"]
        print(self._dataset)

    # LVIS has 1,203 categories - here are some examples
    # LVIS_CATEGORIES = {
    #     0: "aerosol_can", 1: "air_conditioner", 2: "airplane", 3: "alarm_clock",
    #     4: "alcohol", 5: "alligator", 6: "almond", 7: "ambulance", 8: "amplifier",
    #     # ... (the dataset has all 1,203 categories)
    #     1202: "zucchini"
    # }

    def training_prompt(self, sample, detection_ratio=0.5):
        """Create training examples that mix detection and segmentation tasks"""
        
        image = sample["image"]
        ground_truth = sample["ground_truth"]
        objects = sample['objects']
        bboxes = objects['bboxes']
        classes = objects['classes']
        
        # Randomly decide which objects get boxes vs. segmentation
        detection_objects = []
        segmentation_objects = []
        
        for bbox, class_id in zip(bboxes, classes):
            category_name = LVIS_CATEGORIES.get(class_id, f"object_{class_id}")
            
            # Randomly assign tasks
            if random.random() < detection_ratio:
                # This object gets a bounding box
                detection_objects.append({
                    'bbox': bbox[0] if isinstance(bbox[0], list) else bbox,
                    'label': category_name
                })
            else:
                # This object gets segmented
                segmentation_objects.append({
                    'label': category_name
                })
        
        # Create the instruction
        prompt_parts = []
        if detection_objects:
            detection_labels = [obj['label'] for obj in detection_objects]
            prompt_parts.append(f"Put bounding boxes around: {', '.join(detection_labels)}")
        
        if segmentation_objects:
            segment_labels = [obj['label'] for obj in segmentation_objects]
            prompt_parts.append(f"Segment these objects: {', '.join(segment_labels)}")
        
        prompt = " and ".join(prompt_parts) + ". Return results in JSON format."
        
        # Create the expected answer
        response = {
            "bounding_boxes": [
                {"bbox_2d": obj['bbox'], "label": obj['label']} 
                for obj in detection_objects
            ],
            "segmentation_targets": [
                {"label": obj['label'], "task": "segment"} 
                for obj in segmentation_objects
            ]
        }
        
        return prompt, json.dumps(response)

    # Format samples for training
    def format_sample_for_training(self, sample):
        """Convert LVIS sample to Qwen training format"""
        
        prompt, response = training_prompt(sample)
        
        # Create conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            },
            {
                "role": "assistant", 
                "content": response
            }
        ]
        
        return {
            "messages": conversation,
            "image": sample['image']
        }

    # Prepare training data
    print("Preparing training data...")
    train_subset = lvis_dataset['train'].select(range(1000))  # Use 1000 samples
    eval_subset = lvis_dataset['validation'].select(range(200))  # Use 200 for evaluation

    formatted_train = train_subset.map(format_sample_for_training)
    formatted_eval = eval_subset.map(format_sample_for_training)

    print(f"Training samples ready: {len(formatted_train)}")
    print(f"Evaluation samples ready: {len(formatted_eval)}")