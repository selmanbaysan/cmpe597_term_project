import os
from model import DenoisingAutoEncoderModel


def load_train_sentences(folder):
    with open(f"{folder}/unsupervised/train.txt", 'r') as file:
        sentences = file.read().split("\n")
    return sentences

def train_models(epochs=10, batch_size=128, model_name="bert-base-uncased"):
    folders = os.listdir("data-train")
    denoising_functions = ["delete", "noun_swap", "synonym_replacement"]
    model_types = ["custom", "default"]
    for folder in folders:
        #check if folder is a directory
        if os.path.isdir(f"data-train/{folder}"):
            sentences = load_train_sentences(f"data-train/{folder}")
            print(f"Loaded {len(sentences)} sentences from {folder}")
            for model_type in model_types:
                for denoising_function in denoising_functions:
                    print(f"Training DenoisingAutoEncoderModel for {denoising_function} on {folder}")
                    if model_type == "custom":
                        model = DenoisingAutoEncoderModel(model_name, pooling_model="custom", load_model=False)
                    else:
                        model = DenoisingAutoEncoderModel(model_name, load_model=False)
                    model.train(sentences, denoising_function, epochs=epochs, batch_size=batch_size, output_path=f"models/{folder}/{model_type}/{denoising_function}")
                    model.save_model(f"models/{folder}/{model_type}/{denoising_function}")
                    print(f"Training completed for {denoising_function} on {folder}")
            print(f"Training completed for {folder}")
    print("Training completed for all folders")

if __name__ == "__main__":
    train_models(epochs=5, batch_size=128, model_name="bert-base-uncased")
    