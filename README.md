# Fine-tune LLaVa for JSON Data Extraction

This project demonstrates fine-tuning the [LLaVa](https://huggingface.co/docs/transformers/main/en/model_doc/llava) model for extracting structured JSON data from receipt images. The model is trained on the [CORD](https://huggingface.co/datasets/naver-clova-ix/cord-v2) dataset to generate JSON representations of receipt contents, including menu items, prices, subtotals, and totals.

## Features

- **Multimodal Fine-tuning**: Combines vision and language models to extract text and structure from images.
- **Efficient Training**: Uses QLoRa (Quantized Low-Rank Adaptation) for memory-efficient fine-tuning.
- **Modular Codebase**: Organized into separate modules for data handling, model definition, and training.
- **Evaluation Metrics**: Tracks training loss and normalized edit distance for validation.
- **Hugging Face Integration**: Automatic model pushing to the Hub during training.

## Project Structure

```
.
├── config.py                 # Configuration variables and hyperparameters
├── data/
│   └── dataset.py           # Custom dataset class and collate functions
├── model/
│   └── model.py             # PyTorch Lightning module for training
├── utils/
│   └── callbacks.py         # Training callbacks for Hub integration
├── train.py                 # Main training script
├── requirements.txt         # Python dependencies
├── llava-json-data-extractions.ipynb  # Original Jupyter notebook
├── training.png             # Training visualization (if applicable)
└── README.md                # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/VisionLanguageModels-finetuning.git
   cd VisionLanguageModels-finetuning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Hugging Face authentication (for model uploading):
   ```python
   from huggingface_hub import login
   login(token="YOUR_HF_TOKEN")
   ```

## Usage

### Training

Run the main training script:

```bash
python train.py
```

The script will:
- Load the CORD dataset
- Prepare the LLaVa model with QLoRa adapters
- Train on receipt images to generate JSON extractions
- Push checkpoints to Hugging Face Hub

### Configuration

Modify `config.py` to adjust:
- Model parameters (MODEL_ID, REPO_ID)
- Training hyperparameters (batch size, learning rate, epochs)
- LoRa settings (r, alpha, dropout)

## Dataset

The project uses the [CORD dataset](https://huggingface.co/datasets/naver-clova-ix/cord-v2), which contains:
- **Train**: 800 receipt images
- **Validation**: 100 receipt images
- **Test**: 100 receipt images

Each sample includes an image and ground-truth JSON with structured receipt data.

## Model Architecture

- **Base Model**: LLaVa-1.5-7B
- **Quantization**: 4-bit quantization for efficiency
- **Adaptation**: LoRa applied to language model layers (excluding vision components)
- **Task**: Instruction-tuned for JSON extraction from images

## Evaluation

The model is evaluated using normalized edit distance between predicted and ground-truth JSON token sequences. Lower values indicate better performance.

## Results

After training, the model can extract structured data from receipt images, such as:

```json
{
  "menu": [
    {"nm": "Nasi Campur Bali", "cnt": "1 x", "price": "75,000"},
    {"nm": "MilkShake Starwb", "cnt": "1 x", "price": "37,000"}
  ],
  "sub_total": {
    "subtotal_price": "1,346,000",
    "service_price": "100,950",
    "tax_price": "144,695"
  },
  "total": {"total_price": "1,591,600"}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LLaVa](https://huggingface.co/docs/transformers/main/en/model_doc/llava) for the base model
- [CORD Dataset](https://huggingface.co/datasets/naver-clova-ix/cord-v2) for training data
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the implementation
- [PEFT](https://github.com/huggingface/peft) for efficient fine-tuning techniques