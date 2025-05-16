# Img2GPT: Image Captioning with Vision Transformer and GPT-2

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This project demonstrates how to build an end-to-end image captioning model by connecting a pre-trained Vision Transformer (ViT) with a pre-trained GPT-2 language model. It leverages modern techniques like Parameter-Efficient Fine-Tuning (PEFT) with QLoRA for memory-efficient training on consumer GPUs.

## Key Features

- **Hybrid Vision-Language Architecture:** Uses ViT for image understanding and GPT-2 for text generation.
- **Efficient Fine-Tuning:** Employs **QLoRA** (Quantized Low-Rank Adaptation) via the `peft` library to fine-tune both the vision and language models with significantly reduced memory and computational requirements.
- **Conditional Quantization:** The model is trained using 4-bit quantization on GPUs for maximum efficiency, with a fallback to standard precision on CPUs.
- **End-to-End Pipeline:** Includes scripts for data preprocessing, training with periodic evaluation, and a standalone inference script to generate captions for new images.
- **Dataset:** Utilizes the popular **Flickr30k** dataset for training and evaluation.

## Model Architecture

The model works by converting an image into a sequence of embeddings that can be understood by the language model. This sequence then acts as a "prefix" or "prompt" for GPT-2, which generates a descriptive caption.

```
+-----------+     +-----------------+     +--------------------+     +-----------------+     +---------------------+
|   Image   | --> |   ViT Encoder   | --> |  Projection Layer  | --> |  GPT-2 Decoder  | --> |  Generated Caption  |
+-----------+     +-----------------+     +--------------------+     +-----------------+     +---------------------+
   (224x224)      (Outputs patch     (Matches ViT output      (Takes prefix and       ("A photo of a dog...")
                  embeddings)           to GPT-2 input dim)      generates text)
```

## Requirements

To run this project, you'll need Python 3.9+ and the following libraries. You can install them using pip:

```bash
pip install torch torchvision
pip install transformers accelerate bitsandbytes peft
pip install kagglehub pandas tqdm requests
```

Alternatively, you can create a `requirements.txt` file with the following content and run `pip install -r requirements.txt`:

```text
# requirements.txt
torch
torchvision
transformers
accelerate
bitsandbytes
peft
kagglehub
pandas
tqdm
requests
Pillow
```
**Note:** `bitsandbytes` for 4-bit quantization requires a CUDA-enabled GPU.

## How to Use

### 1. Training

The `training_script.py` handles the entire training process from data download to model saving.

**Configuration:**
Before running, you can adjust the following parameters at the top of the script:
- `DEVICE_CHOICE`: Automatically set to `"cuda"` if available, otherwise `"cpu"`.
- `BATCH_SIZE`: Adjust based on your GPU memory (e.g., 16, 20).
- `NUM_EPOCHS`: Number of epochs to train for (e.g., 3).
- `EVAL_INTERVAL`: How often (in batches) to generate sample captions for evaluation.

**Running the Training:**
Execute the script in your environment (e.g., a Colab notebook cell).

```bash
python training_script.py
```

**Output:**
- The script will download the Flickr30k dataset via KaggleHub.
- It will print the training loss for each batch and an average loss at the end of each epoch.
- Every `EVAL_INTERVAL` batches, it will print sample generated captions for a fixed set of images to show learning progress.
- A model weights file (e.g., `img2gpt_epoch_1.pth`) will be saved after each epoch.

### 2. Inference

The `inference_script.py` is a standalone script to load your trained model and generate a caption for any image from a URL.

**Setup:**
1.  **Place your trained weights file** in the same directory as the script.
2.  Open `inference_script.py` and modify the following variables:
    - `MODEL_WEIGHTS_PATH`: Set this to the name of your saved `.pth` file (e.g., `"img2gpt_epoch_3.pth"`).
    - `IMAGE_URL`: Provide a direct URL to the image you want to caption.

**Running Inference:**
Execute the script from your terminal.

```bash
python inference_script.py
```

**Output:**
The script will reconstruct the model, load your trained weights, download and preprocess the image, and finally print the generated caption.

```
$ python inference_script.py
Using device: cuda
Step 1: Loading tokenizer and base models (with 4-bit quantization)...
Step 2: Applying the exact same LoRA configurations...
Step 3: Initializing the combined Img2GPT model...
Step 4: Loading trained weights from 'img2gpt_epoch_3.pth'...
Model successfully built and ready for inference!

Preprocessing image from URL: http://images.cocodataset.org/val2017/000000039769.jpg
Generating caption...

==================================================
      INFERENCE RESULT
==================================================
Image URL: http://images.cocodataset.org/val2017/000000039769.jpg
Generated Caption: a group of cats are sitting on a couch
==================================================
```

## Future Improvements

- **Use Larger Models:** Replace `gpt2` with more powerful models like `Llama-3-8B` or `Mistral-7B` for higher-quality captions.
- **More Advanced Prompting:** Implement a more sophisticated prompt structure instead of a simple linear projection.
- **Web Interface:** Build a simple Gradio or Streamlit app to provide a user-friendly interface for captioning images.
- **Broader Dataset:** Train on a larger, more diverse dataset like COCO or LAION to improve generalization.

## Acknowledgements

- This project is heavily reliant on the amazing work by **Hugging Face** for their `transformers` and `peft` libraries.
- The **Flickr30k dataset** provided by [hsankesara on Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset).
- The **QLoRA** technique introduced in the paper [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.