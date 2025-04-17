import sys
import traceback

try:
    from datasets import load_dataset, load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Model loading params
    load_in_4bit = True

    # LoRA Params
    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 16
    lora_bias = "all"
    model_type = "deepseek"
    dataset_type = "psychology"
    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    # Trainer params
    output_dir = "outputs_psychology"
    optim_type = "adafactor"
    learning_rate = 0.00005
    weight_decay = 0.002
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8
    gradient_accumulation_steps = 2
    warmup_steps = 5
    save_steps = 100
    logging_steps = 25

    # Load in the model as a 4-bit or 8-bit model
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "WizardLM/WizardLM-13B-V1.2" if model_type == "wizard13"
            else "TheBloke/wizardLM-7B-HF" if model_type == "wizard7"
            else "tiiuae/falcon-7b" if model_type == "falcon"
            else "deepseek-ai/DeepSeek-V3-0324" if model_type == "deepseek"
            else "meta-llama/Llama-2-7b-hf",
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_config,
            cache_dir="./models",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            "WizardLM/WizardLM-13B-V1.2" if model_type == "wizard13"
            else "TheBloke/wizardLM-7B-HF" if model_type == "wizard7"
            else "tiiuae/falcon-7b" if model_type == "falcon"
            else "deepseek-ai/DeepSeek-V3-0324" if model_type == "deepseek"
            else "meta-llama/Llama-2-7b-hf",
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit=True,
            cache_dir="./models",
        )

    # Load in the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "WizardLM/WizardLM-13B-V1.2" if model_type == "wizard13"
        else "TheBloke/wizardLM-7B-HF" if model_type == "wizard7"
        else "tiiuae/falcon-7b" if model_type == "falcon"
        else "deepseek-ai/DeepSeek-V3-0324" if model_type == "deepseek"
        else "meta-llama/Llama-2-7b-hf",
        trust_remote_code=True,
        cache_dir="./models",
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load in the dataset and map using the tokenizer
    if dataset_type == "psychology":
        dataset = load_dataset(
            "samhog/psychology-6k",
            cache_dir="./datasets",
        )

        # Load in the dataset and map using the tokenizer
        def map_function(example):
            text = example["text"]
            # Encode the question and output
            text_encoded = tokenizer(text, max_length=max_length-1, truncation=True, padding="max_length")
            return text_encoded

    # Continue with the rest of your code...

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Traceback:")
    traceback.print_exc(file=sys.stdout)
