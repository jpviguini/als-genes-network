import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    LineByLineTextDataset,
)

BASE_MODEL = "dmis-lab/biobert-large-cased-v1.1"
OUTPUT_DIR = "./biobert_LARGE_als_adapted_model"
DATA_FILE = "../data/corpus_als_general_pmc_preprocessed3.csv"
EPOCHS = 3
BATCH_SIZE = 16
MAX_LEN = 256

TB_LOG_DIR = "./tb_mlm_logs"  # tensorboard logs 
LOGGING_STEPS = 200         

def train_mlm():
    # gpu check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    if torch.cuda.is_available():
        print(f"GPU Details: {torch.cuda.get_device_name(0)}")

    print(f"Loading tokenizer and model from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)

    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE, escapechar="\\")
    texts = df["text"].astype(str).tolist()

    temp_text_file = "temp_corpus_for_training.txt"
    print("Preparing text file for dataset...")
    with open(temp_text_file, "w", encoding="utf-8") as f:
        for t in tqdm(texts, desc="Writing lines"):
            f.write(t + "\n")

    print("Tokenizing data...")
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=temp_text_file,
        block_size=MAX_LEN,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./results_checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=10_000,
        save_total_limit=2,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        disable_tqdm=False,
        # tensorboard logging
        report_to=["tensorboard"],
        logging_dir=TB_LOG_DIR,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        log_level="info",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("Starting Training (MLM)...")
    trainer.train()

    print(f"Saving adapted model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    if os.path.exists(temp_text_file):
        os.remove(temp_text_file)

    print("Done!")
    print(f"TensorBoard logs in: {TB_LOG_DIR}")
    print(f"Run: tensorboard --logdir {TB_LOG_DIR} --host 0.0.0.0 --port 6006")


if __name__ == "__main__":
    train_mlm()
