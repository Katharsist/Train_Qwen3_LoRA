from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
import argparse
import torch
import config

def load_model(use_lora: bool):
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
        base_model = AutoModelForCausalLM.from_pretrained(config.MODEL_PATH)
        if use_lora:
            print("[*] Загрузка Qwen3 с LoRA")
            model = PeftModel.from_pretrained(base_model, config.LORA_PATH)
        else:
            print("[*] Загрузка Qwen3")
            model = base_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки модели: {e}")

def main():
    parser = argparse.ArgumentParser(description="Тестирование модели")
    parser.add_argument("--seed", type=int, default=None, help="Использование сида (число между 0 и 2**32 - 1)")
    parser.add_argument("--lora", action="store_true", help="Использование LoRA")

    args = parser.parse_args()
    if args.seed is not None:
        print(f"[*] Seed = {args.seed}")
        set_seed(args.seed)
    tokenizer, model = load_model(use_lora=args.lora)
    prompt = config.PROMPT
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=config.GEN_MAX_NEW_TOKENS,
        temperature=config.GEN_TEMPERATURE,
        top_p=config.GEN_TOP_P,
        do_sample=config.GEN_DO_SAMPLE,
    )
    content = tokenizer.decode(outputs[0], skip_special_tokens=True).strip("\n")

    print("Prompt:", prompt)
    print("Qwen:", content)

if __name__ == "__main__":
    main()