import os
import requests
import torch
from torch.nn import Module
from torchvision import transforms
from torchvision.models.video import r3d_18
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    Wav2Vec2ForCTC,
    T5Tokenizer,
    Wav2Vec2Processor,
)
from torchscale.architecture.config import DecoderConfig
from torchscale.architecture.decoder import Decoder
from torchscale.component.embedding import PositionalEmbedding
import bitsandbytes
from flamingo_pytorch import PerceiverResampler

class BaseTokenizer:
    def tokenize(self, data):
        raise NotImplementedError('This method should be implemented in a subclass')
    
    def process(self, data):
        raise NotImplementedError("This method should be implemented in a subclass")
    
    def embed(self, data):
        raise NotImplementedError("This method should be implemented in a subclass")


class ModalityDetector:
    def __init__(self, method, input_data, user_input=None):
        self.method = method
        self.input_data = input_data
        self.user_input = user_input

    def get_modality(self):
        if self.method == "file_extension":
            return self.detect_modality_from_file_extension()
        elif self.method == "content_based":
            return self.detect_modality_from_content()
        elif self.method == "user_input":
            return self.user_input

    def detect_modality_from_file_extension(self):
        _, file_extension = os.path.splitext(self.input_data)
        file_extension = file_extension.lower()

        if file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
            return 'image'
        elif file_extension in ['.wav', '.mp3', '.ogg']:
            return 'audio'
        elif file_extension in [".txt", '.md', '.json']:
            return 'text'

    def detect_modality_from_content(self):
        pass
    
class TokenizerFactory:
    def create_tokenizer(self, modality):
        # Fetch models from Hugging Face API
        api_url = "https://huggingface.co/api/models"
        response = requests.get(api_url)

        if response.status_code != 200:
            raise ValueError("Failed to fetch models from Hugging Face API")

        models = response.json()

        # Filter models based on modality and sort by likes
        matching_models = sorted(
            [model for model in models if modality in model["tags"]],
            key=lambda x: x["likes"],
            reverse=True
        )

        if not matching_models:
            raise ValueError(f"No matching tokenizer found for modality '{modality}'")

        # Select the most liked tokenizer and instantiate it
        selected_model = matching_models[0]["modelId"]
        tokenizer = AutoTokenizer.from_pretrained(selected_model)

        return tokenizer


class ModalityProcessor:
    def __init__(self, modality_detector):
        self.modality_detector = modality_detector
        self.modalities = {}
        self.tokenizer_factory = TokenizerFactory(self.modality_detector)

    def processor(self, modality, data):
        modality = self.modality_detector.get_modality()

        if modality in self.modalities:
            tokenizer = self.modalities[modality]
        else:
            tokenizer = self.tokenizer_factory.create_tokenizer(modality)
            self.modalities[modality] = tokenizer

        tokens = tokenizer(data, return_tensors="pt", padding=True, truncation=True)

        return tokens
    

class KosmosEmbedder(torch.nn.Module):
    def __init__(self, modality):
        super().__init__()
        self.modality = modality 
        self.tokenizer = AutoTokenizer.from_pretrained(modality)
        self.model = AutoModel.from_pretrained(modality)
        self.proj = torch.nn.Linear(self.model.config.hidden_size, 2048)

    def forward(self, data):
        tokens = self.tokenizer(data, return_tensors="pt", padding=True, truncation=True)
        output = self.model(**tokens)
        embed = self.proj(output.last_hidden_state)

        return embed


class KosmosTokenizer:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
        self.audio_tokenizer = Wav2Vec2Processor.from_pretrained("facebook/data2vec-audio-base-960h")
        self.tokenizer = T5Tokenizer.from_pretrained(
            "t5-large",
            additional_special_tokens=["<image>", "</image>", "<audio>", "</audio>", "<video>", "</video>", "<any>", "</any>"],
            extra_ids=0,
            model_max_length=1984
        )
        self.video_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        ])
        self.vid_idx, self.vid_end_ix = self.tokenizer.convert_tokens_to_ids(["<video>", "</video>"])
        self.audio_idx, self.audio_end_idx = self.tokenizer.convert_tokens_to_ids(["<audio>", "</audio>"])
        self.im_idx, self.im_end_idx = self.tokenizer.convert_tokens_to_ids(["<image>", "</image>"])
        self.any_idx, self.any_end_idx = self.tokenizer.convert_tokens_to_ids(["<any>", "</any>"])

    def tokenize_texts(self, texts):
        texts =  self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids
        media_tokens = torch.tensor([[self.im_idx, self.im_end_idx, self.audio_idx, self.audio_end_idx, self.vid_idx, self.vid_end_idx, self.any_idx, self.any_end_idx]] * texts.shape[0])
        return torch.cat([texts[:, 0:1], media_tokens, texts[:, 1:]], dim=1), texts
    
    def tokenize_images(self, images):
        return self.processor(images=images, return_tensors="pt").pixel_values
    
    def tokenize_audio(self, audios):
        return self.audio_tokenizer(audios, return_tensors="pt", padding=True, truncation=True).input_values
    
    def tokenize_videos(self, videos):
        if not videos:
            return None
        processed_videos = []
        for video in videos:
            video_frames = [self.video_transform(frame) for frame in video]
            processed_videos.append(torch.stack(video_frames))
        return torch.stack(processed_videos)
    
    def tokenize(self, sample):
        text_tokens, only_text_tokens = self.tokenize_texts(sample["target_text"])
        attention_mask = text_tokens != self.tokenizer.pad_token_id
        dummy_image_features = torch.ones((text_tokens.shape[0], 64))
        attention_mask = torch.cat([dummy_image_features, attention_mask], dim=1)
        return {
            "text_tokens": text_tokens,
            "images": self.tokenize_images(sample["image"]),
            "labels": only_text_tokens,
            "attention_mask": attention_mask,
            "audios": self.tokenize_audio(sample["audio"]),
            "videos": self.tokenize_videos(sample["video"])
        }

class Kosmos(Module):
    def __init__(self, modality, modality_detector):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K").vision_model
        self.audio_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.video_model = r3d_18(pretrained=True)
        self.video_model = torch.nn.Sequential(*list(self.video_model.children())[:-1])
        self.modality_detector = modality_detector
        self.tokenizer = KosmosTokenizer()
        self.processor = ModalityProcessor(modality_detector)
        self.embedder = KosmosEmbedder(modality)

        self.embed = bitsandbytes.nn.modules.Embedding(
            32002,
            2048,
            padding_idx=1
        )
        self.embed_positions= PositionalEmbedding(
            2048,
            2048,
            1
        )

        self.output_projection = torch.nn.Linear(
            2048, 32002, bias=False
        )
        torch.nn.init.normal_(
            self.output_projection.weight, mean=0, std=2048**-0.5
        )

        self.config = DecoderConfig(
            decoder_layers=24,
            decoder_embed_dim=2048,
            decoder_ffn_embed_dim=8192,
            decoder_attention_heads=32,
            dropout=0.1,
            activation_fn="gelu",
            attention_dropout=0.1,
            vocab_size=64007,
            subln=True,
            xpos_rel_pos=True,
            max_rel_pos=2048
        )
        self.decoder = Decoder(
            self.config,
            embed_tokens=self.embed,
            embed_positions=self.embed_positions,
            output_projection=self.output_projection
        )

        self.perceive = PerceiverResampler(
            dim = 1024,
            depth = 2,
            dim_head = 64,
            heads = 8,
            num_latents = 64,
            num_media_embeds = 257
        )

        self.image_proj = torch.nn.Linear(1024, 2048, bias=False)
        torch.nn.init.normal_(
            self.image_proj.weight, mean=0, std=2048**-0.5
        )

        self.audio_proj = torch.nn.Linear(768, 2048, bias=False)
        torch.nn.init.normal_(
            self.audio_proj.weight, mean=0, std=2048 ** -0.5
        )

        self.video_proj = torch.nn.Linear(512, 2048, bias=False)
        torch.nn.init.normal_(
            self.video_proj.weight, mean=0, std=2048 ** -0.5
        )

    def forward(self, text_tokens, images, audios, videos, any_modality, **kwargs):
        images = self.clip_model(pixel_values=images)["last_hidden_state"]
        images = self.perceive(images).squeeze(1)
        images = self.image_proj(images)

        audios = self.audio_model(audios).logits
        audios = audios.mean(dim=1)
        audios = self.audio_proj(audios)

        if videos is not None:
            videos = videos.transpose(1, 2).contiguous()
            videos = self.video_model(videos)
            videos = videos.view(videos.size(0), -1)
            videos = self.video_proj(videos)

        any_embeddings = []
        for modality_data in any_modality:
            modality = modality_data["modality"]
            data = modality_data["data"]
            tokens = self.processor.processor(modality, data)
            embed = self.embedder(modality)(tokens)
            any_embeddings.append(embed)
        any_embeddings = torch.stack(any_embeddings)

        model_input = self.decoder.forward_embedding(text_tokens)[1]
        model_input = torch.cat([model_input[:, 0:6], images, audios, videos, any_embeddings, model_input[:, 6:]], dim=1)
        model_input = self.decoder.forward_embedding(model_input, token_embedding=model_input)[0]

        return self.decoder(model_input, passed_x=model_input)[0]