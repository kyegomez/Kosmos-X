import torch
from torchscale.architecture.config import DecoderConfig
from torchscale.architecture.decoder import Decoder
from torchscale.component.embedding import PositionalEmbedding
from transformers import T5Tokenizer, CLIPProcessor, CLIPModel
from transformers import Wav2Vec2Tokenizer
from transformers import Wav2Vec2Model

from flamingo_pytorch import PerceiverResampler
from torch.nn import Module
import bitsandbytes


class KosmosTokenizer:
    def __init__(self, modalities=["text", "image", "audio"]):
        self.modalities = modalities
        self.processor = CLIPProcessor.from_pretrained(
            "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
        )

        # T5 uses SentencePiece tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(
            "t5-large",
            additional_special_tokens=[
                "<image>",
                "</image>",
                "<audio>",
                "</audio>",
            ],
            extra_ids=0,
            model_max_length=1984,
        )
        self.audio_idx, self.audio_end_idx = (
            self.tokenizer.convert_tokens_to_ids(["<audio>", "</audio>"])
        )
        self.im_idx, self.im_end_idx = self.tokenizer.convert_tokens_to_ids(
            ["<image>", "</image>"]
        )
        self.audio_tokenizer = Wav2Vec2Tokenizer.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

    def tokenize_texts(self, texts):
        texts = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).input_ids
        # Add image and audio tokens to text as "<s> <image> </image> <audio> </audio> text </s>"
        media_tokens = torch.tensor(
            [[self.im_idx, self.im_end_idx, self.audio_idx, self.audio_end_idx]]
            * texts.shape[0]
        )
        return (
            torch.cat([texts[:, 0:1], media_tokens, texts[:, 1:]], dim=1),
            texts,
        )

    def tokenize_images(self, images):
        return self.processor(images=images, return_tensors="pt").pixel_values

    def tokenize_audio(self, audios):
        return self.audio_tokenizer(
            audios, return_tensors="pt", padding=True, truncation=True
        ).input_ids

    def tokenize(self, sample):
        text_tokens, only_text_tokens = self.tokenize_texts(
            sample["target_text"]
        )
        attention_mask = text_tokens != self.tokenizer.pad_token_id

        if "image" in self.modalities:
            images = self.tokenize_images(sample["image"])
        else:
            images = None

        if "audio" in self.modalities:
            audios = self.tokenize_audio(sample["audio"])
        else:
            audios = None

        return {
            "text_tokens": text_tokens,
            "images": images,
            "labels": only_text_tokens,
            "attention_mask": attention_mask,
            "audios": audios,
        }


class Kosmos(Module):
    def __init__(self, modalities=["text", "image", "audio"]):
        super().__init__()
        # Instantiate Clip Vit-l/14
        self.modalities = modalities
        self.clip_model = CLIPModel.from_pretrained(
            "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
        ).vision_model
        self.audio_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

        self.embed = bitsandbytes.nn.modules.Embedding(
            32002, 2048, padding_idx=1
        )
        self.embed_positions = PositionalEmbedding(2048, 2048, 1)

        self.output_projection = torch.nn.Linear(2048, 32002, bias=False)
        torch.nn.init.normal_(
            self.output_projection.weight, mean=0, std=2048**-0.5
        )

        # Config following KOSMOS-1 paper (https://arxiv.org/pdf/2302.14045.pdf)
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
            max_rel_pos=2048,
        )
        self.decoder = Decoder(
            self.config,
            embed_tokens=self.embed,
            embed_positions=self.embed_positions,
            output_projection=self.output_projection,
        )

        self.perceive = PerceiverResampler(
            dim=1024,
            depth=2,
            dim_head=64,
            heads=8,
            num_latents=64,
            num_media_embeds=257,
        )

        self.image_proj = torch.nn.Linear(1024, 2048, bias=False)
        torch.nn.init.normal_(self.image_proj.weight, mean=0, std=2048**-0.5)

        # add audio
        self.audio_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        self.audio_proj = torch.nn.Linear(768, 2048, bias=False)
        torch.nn.init.normal_(self.audio_proj.weight, mean=0, std=2048**-0.5)

    def forward(self, text_tokens, images, audios, **kwargs):
        if "image" in self.modalities:
            images = self.clip_model(pixel_values=images)["last_hidden_state"]
            images = self.perceive(images).squeeze(1)
            images = self.image_proj(images)

        if "audio" in self.modalities:
            audios = self.audio_model(input_ids=audios).last_hidden_state
            audios = audios.mean(dim=1)
            audios = self.audio_proj(audios)

        model_input = self.decoder.forward_embedding(text_tokens)[1]
        model_input = torch.cat(
            [model_input[:, 0:3], images, audios, model_input[:, 3:]], dim=1
        )
        model_input = self.decoder.forward_embedding(
            model_input, token_embedding=model_input
        )[0]

        return self.decoder(model_input, passed_x=model_input)[0]
