from sentence_transformers import SentenceTransformer, models, util, datasets, evaluation, losses
import torch
from torch import nn
import os
from DenoisingAutoEncoderDataset import DenoisingAutoEncoderDataset
from sentence_transformers import losses
from torch.utils.data import DataLoader
import nltk

nltk.download("punkt")


class DynamicKMaxPooling(nn.Module):
    def __init__(self, top_k, L):
        super(DynamicKMaxPooling, self).__init__()
        self.top_k = top_k
        self.L = L

    def forward(self, x, layer_idx):
        max_k = x.size(2)  # Get the current size of the dimension to apply topk
        k = max(self.top_k, int((self.L - layer_idx) / self.L * max_k))
        k = min(k, max_k)  # Ensure k is not larger than the actual dimension size
        print(f"Layer Index: {layer_idx}, Calculated k: {k}, Max k: {max_k}, Current x.size(): {x.size()}")
        return x.topk(k, dim=2)[0]


class CNNPoolingModule(nn.Module):
    def __init__(self, in_channels, layers_info, output_dim, is_dynamic):
        super(CNNPoolingModule, self).__init__()
        self.conv_blocks = nn.ModuleList()
        self.is_dynamic = is_dynamic
        self.L = len(layers_info)
        current_channels = in_channels
        self.final_dimensions = in_channels

        for layer_depth, (num_filters, kernel_size, use_pooling, top_k) in enumerate(layers_info):
            if self.is_dynamic and layer_depth == len(layers_info) - 1:
                conv_block = nn.Sequential(
                    nn.Conv1d(in_channels=current_channels,
                              out_channels=768,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2),  # padding to maintain dimensionality
                    nn.BatchNorm1d(768),
                    nn.ReLU()
                )
                conv_block.add_module('pooling', nn.AdaptiveMaxPool1d(1))
                self.final_dimensions = num_filters  # Only the number of filters matters if length is 1

            else:
                conv_block = nn.Sequential(
                    nn.Conv1d(in_channels=current_channels,
                              out_channels=num_filters,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2),  # padding to maintain dimensionality
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU()
                )
                if use_pooling:
                    if self.is_dynamic:
                        # Apply dynamic k-max pooling
                        pooling_layer = DynamicKMaxPooling(top_k, self.L)
                        conv_block.add_module('dynamic_pooling', pooling_layer)
                        self.final_dimensions = num_filters * top_k  # Adjust for dynamic k
                    else:
                        # Apply static pooling
                        conv_block.add_module('pooling', nn.AdaptiveMaxPool1d(1))
                        self.final_dimensions = num_filters  # Only the number of filters matters if length is 1

            self.conv_blocks.append(conv_block)
            current_channels = num_filters  # Update the channel number for the next layer

        # The final layer's output is pooled to produce a single vector if the last layer isn't pooled
        self.final_pooling = nn.Identity() if layers_info[-1][2] else nn.AdaptiveMaxPool1d(1)
        self.final_dimensions = 768 if self.is_dynamic else current_channels
        self.output_layer = nn.Linear(self.final_dimensions, output_dim)

    def forward(self, features):
        x = features["token_embeddings"]
        x = x.permute(0, 2, 1)  # Prepare for Conv1d
        for idx, block in enumerate(self.conv_blocks):
            if self.is_dynamic and idx != len(self.conv_blocks)-1:
                x = block[:-1](x)  # Apply all layers except the last dynamic pooling
                x = block[-1](x, idx)  # Apply dynamic pooling with layer index
            else:
                x = block(x)
        x = self.final_pooling(x).squeeze(2)  # Remove the last dimension after pooling
        x = x.view(x.size(0), -1)
        output = self.output_layer(x)
        features.update({"sentence_embedding": output})
        return features

    def save(self, output_path):
        new_output_path = os.path.join(output_path, "model.pth")
        torch.save(self.state_dict(), new_output_path)

    def load(self, input_path):
        self.load_state_dict(torch.load(input_path))


def calculate_dynamic_k(layer_idx, L, max_seq_length, k_end=3):
    k_start = max(int(max_seq_length / 3), k_end + L)  # Ensure k_start is reasonable
    min_decrement = (k_start - k_end) / (L - 1) if L > 1 else 0  # Avoid division by zero

    k = max(k_end, int(k_start - min_decrement * layer_idx))
    return k


class DenoisingAutoEncoderModel:
    def __init__(self, transformer_model, pooling_model=None, load_model=False, max_seq_length=50):
        self.transformer_model = transformer_model
        # calculate dynamic top_k values -> higher in first layers, gets lower towards last layers
        self.layers_info = [
            (128, 3, True, calculate_dynamic_k(0, 4, max_seq_length)),  # 128 filters, kernel size 3, pooling, top_k
            (128, 5, True, calculate_dynamic_k(1, 4, max_seq_length)),  # 128 filters, kernel size 5, pooling, top_k
            (256, 3, True, calculate_dynamic_k(2, 4, max_seq_length)),  # 256 filters, kernel size 3, pooling, top_k
            (256, 5, True, calculate_dynamic_k(3, 4, max_seq_length))  # 256 filters, kernel size 5, pooling, top_k
        ]
        if load_model:
            self.model = self.load_model(self.transformer_model, pooling_model)
        else:
            self.model = self.create_model(self.transformer_model, pooling_model)

    def create_model(self, transformer_model, pooling_model=None):
        word_embedding_model = models.Transformer(transformer_model)
        if pooling_model is None:
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), "cls")

        elif pooling_model == "custom":
            pooling_model = CNNPoolingModule(in_channels=word_embedding_model.get_word_embedding_dimension(),
                                             layers_info=self.layers_info, output_dim=768, is_dynamic=False)

        elif pooling_model == "dynamic":
            pooling_model = CNNPoolingModule(in_channels=word_embedding_model.get_word_embedding_dimension(),
                                             layers_info=self.layers_info, output_dim=768, is_dynamic=True)

        return SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def load_model(self, transformer_model, pooling_model):
        if pooling_model is None:
            return SentenceTransformer(transformer_model)

        elif pooling_model == "custom":
            model = models.Transformer(transformer_model)
            cnn_module = CNNPoolingModule(in_channels=model.get_word_embedding_dimension(),
                                          layers_info=self.layers_info, output_dim=768, is_dynamic=False)
            cnn_module.load_state_dict(torch.load(f'{transformer_model}/1_CNNPoolingModule/model.pth'))
            return SentenceTransformer(modules=[model, cnn_module])

        elif pooling_model == "dynamic":
            model = models.Transformer(transformer_model)
            cnn_module = CNNPoolingModule(in_channels=model.get_word_embedding_dimension(),
                                          layers_info=self.layers_info, output_dim=768, is_dynamic=True)
            cnn_module.load_state_dict(torch.load(f'{transformer_model}/1_CNNPoolingModule/model.pth'))
            return SentenceTransformer(modules=[model, cnn_module])

    def save_model(self, output_path):
        self.model.save(output_path)

    def train(self, sentences, denoising_function, epochs=10, batch_size=128, output_path="models"):
        if denoising_function == "delete":
            dataset = DenoisingAutoEncoderDataset(sentences, noise_fn=DenoisingAutoEncoderDataset.delete)
        elif denoising_function == "noun_swap":
            dataset = DenoisingAutoEncoderDataset(sentences, noise_fn=DenoisingAutoEncoderDataset.noun_swap)
        elif denoising_function == "synonym_replacement":
            dataset = DenoisingAutoEncoderDataset(sentences, noise_fn=DenoisingAutoEncoderDataset.synonym_replacement)
        else:
            raise ValueError("Invalid denoising function")

        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        train_loss = losses.DenoisingAutoEncoderLoss(self.model, decoder_name_or_path=self.transformer_model,
                                                     tie_encoder_decoder=False)

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            scheduler="constantlr",
            optimizer_params={"lr": 3e-5},
            output_path=output_path,
            show_progress_bar=True,
        )
        del train_dataloader, train_loss

    def print_my_model(self):
        print(self.model)
