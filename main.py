import os
from model import DenoisingAutoEncoderModel
import gc
import numpy as np


def load_train_sentences(folder):
    with open(f"{folder}/unsupervised/train.txt", 'r', encoding="utf-8") as file:
        sentences = file.read().split("\n")
    return sentences


def train_models(epochs=10, batch_size=128, model_name="bert-base-uncased"):
    folders = ["askubuntu", "cqadupstack"]
    denoising_functions = ["delete", "noun_swap", "synonym_replacement"]
    model_types = ["dynamic"]
    for folder in folders:
        # check if folder is a directory
        if os.path.isdir(f"data-train/{folder}"):
            sentences = load_train_sentences(f"data-train/{folder}")
            print(f"Loaded {len(sentences)} sentences from {folder}")

            # find max seq length
            token_lengths = [len(sentence.split()) for sentence in sentences]

            # Calculate percentiles
            percentile_95 = np.percentile(token_lengths, 95)
            percentile_99 = np.percentile(token_lengths, 99)

            print(f"95th percentile of token lengths: {percentile_95}")
            print(f"99th percentile of token lengths: {percentile_99}")

            # Set max sequence length based on percentile
            max_seq_length = int(percentile_95)  # or use percentile_99 if less truncation is desired

            for model_type in model_types:
                for denoising_function in denoising_functions:
                    print(f"Training DenoisingAutoEncoderModel for {denoising_function} on {folder}")
                    if model_type == "custom":
                        model = DenoisingAutoEncoderModel(model_name, pooling_model="custom", load_model=False, max_seq_length=max_seq_length)

                    elif model_type == "dynamic":
                        model = DenoisingAutoEncoderModel(model_name, pooling_model="dynamic", load_model=False, max_seq_length=max_seq_length)

                    else:
                        model = DenoisingAutoEncoderModel(model_name, load_model=False)

                    model.print_my_model()
                    model.train(sentences, denoising_function, epochs=epochs, batch_size=batch_size,
                                output_path=f"models/{folder}/{model_type}/{denoising_function}")
                    model.save_model(f"models/{folder}/{model_type}/{denoising_function}")

                    gc.collect()
                    del model
                    print(f"Training completed for {denoising_function} on {folder}")

            print(f"Training completed for {folder}")
    print("Training completed for all folders")


if __name__ == "__main__":
    train_models(epochs=5, batch_size=2, model_name="bert-base-uncased")
