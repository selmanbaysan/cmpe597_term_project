from sentence_transformers import SentenceTransformer, models, util, datasets, evaluation, losses
import torch
from torch import nn
import os
from DenoisingAutoEncoderDataset import DenoisingAutoEncoderDataset
from sentence_transformers import losses
from torch.utils.data import DataLoader
import nltk
nltk.download("punkt")

class CNNPoolingModule(nn.Module):
    def __init__(self, in_channels, layers_info, output_dim):
        super(CNNPoolingModule, self).__init__()
        self.conv_blocks = nn.ModuleList()
        current_channels = in_channels

        for layer_depth, (num_filters, kernel_size, use_pooling) in enumerate(layers_info):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=current_channels,
                          out_channels=num_filters,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2),  # padding to maintain dimensionality
                nn.BatchNorm1d(num_filters),
                nn.ReLU()
            )
            if use_pooling:
                # Apply pooling at certain layers according to the design (True/False flag)
                conv_block.add_module('pooling', nn.AdaptiveMaxPool1d(1))
            self.conv_blocks.append(conv_block)
            current_channels = num_filters  # Update the channel number for the next layer

        # The final layer's output is pooled to produce a single vector if the last layer isn't pooled
        if not layers_info[-1][2]:
            self.final_pooling = nn.AdaptiveMaxPool1d(1)
        else:
            self.final_pooling = nn.Identity()

        self.output_layer = nn.Linear(current_channels, output_dim)

    def forward(self, features):
        x = features["token_embeddings"]
        x = x.permute(0, 2, 1)  # Prepare for Conv1d
        for block in self.conv_blocks:
            x = block(x)
        x = self.final_pooling(x).squeeze(2)  # Remove the last dimension after pooling
        output = self.output_layer(x)
        features.update({"sentence_embedding": output})
        return features

    def save(self, output_path):
        new_output_path = os.path.join(output_path, "model.pth")
        torch.save(self.state_dict(), new_output_path)

    def load(self, input_path):
        self.load_state_dict(torch.load(input_path))

class DenoisingAutoEncoderModel():
    def __init__(self, transformer_model, pooling_model=None, load_model=False):
        self.transformer_model = transformer_model
        self.layers_info = [
                                (128, 3, False),  # 128 filters, kernel size 3, no pooling
                                (128, 5, False),  # 128 filters, kernel size 5, no pooling
                                (256, 3, False),  # 256 filters, kernel size 3, no pooling
                                (256, 5, True)  # 256 filters, kernel size 5, pooling here
                            ]
        if load_model:
            self.model = self.load_model(self.transformer_model, pooling_model)
        else:
            self.model = self.create_model(self.transformer_model, pooling_model)
        
    
    def create_model(self, transformer_model, pooling_model=None):
        word_embedding_model = models.Transformer(transformer_model)
        if pooling_model is None:
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_cls_token=False, pooling_mode_max_tokens=True)
        elif pooling_model == "custom":
            pooling_model = CNNPoolingModule(in_channels=word_embedding_model.get_word_embedding_dimension(), layers_info=self.layers_info, output_dim=768)

        return SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    def load_model(self, transformer_model, pooling_model):
        if pooling_model is None:
            return SentenceTransformer(transformer_model)
        elif pooling_model == "custom":
            model = models.Transformer(transformer_model)
            cnn_module = CNNPoolingModule(in_channels=model.get_word_embedding_dimension(), layers_info=self.layers_info, output_dim=768)
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
        train_loss = losses.DenoisingAutoEncoderLoss(self.model, decoder_name_or_path=self.transformer_model, tie_encoder_decoder=True)

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            scheduler="constantlr",
            optimizer_params={"lr": 3e-5},
            output_path=output_path,
            show_progress_bar=True,
        )

    

        