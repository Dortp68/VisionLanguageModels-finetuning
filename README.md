# Fine-tune LLaVa for JSON Data Extraction

This project demonstrates fine-tuning the [LLaVa](https://huggingface.co/docs/transformers/main/en/model_doc/llava) model for extracting structured JSON data from receipt images. The model is trained on the [CORD](https://huggingface.co/datasets/naver-clova-ix/cord-v2) dataset to generate JSON representations of receipt contents, including menu items, prices, subtotals, and totals.

## Features

- **Multimodal Fine-tuning**: Combines vision and language models to extract text and structure from images.
- **Efficient Training**: Uses QLoRa (Quantized Low-Rank Adaptation) for memory-efficient fine-tuning.
- **Modular Codebase**: Organized into separate modules for data handling, model definition, and training.
- **Evaluation Metrics**: Tracks training loss and normalized edit distance for validation.
- **Hugging Face Integration**: Automatic model pushing to the Hub during training.

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
![](https://github.com/Dortp68/VisionLanguageModels-finetuning/blob/main/training.png)

Finetuned model: https://huggingface.co/Dortp58/Llava-11B-finetuned
## Acknowledgments

- [LLaVa](https://huggingface.co/docs/transformers/main/en/model_doc/llava) for the base model
- [CORD Dataset](https://huggingface.co/datasets/naver-clova-ix/cord-v2) for training data
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the implementation
- [PEFT](https://github.com/huggingface/peft) for efficient fine-tuning techniques
