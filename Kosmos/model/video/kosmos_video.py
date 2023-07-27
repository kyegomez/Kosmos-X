import torch
from torchscale.architecture.config import DecoderConfig
from torchscale.architecture.decoder import Decoder
from torchscale.component.embedding import PositionalEmbedding
from transformers import T5Tokenizer, CLIPProcessor, CLIPModel
from transformers import Data2VecForCTC, Wav2Vec2Processor

from flamingo_pytorch import PerceiverResampler
from torch.nn import Module
import bitsandbytes

#video
#preprecoess videos and tokenize them -> projection layer to transform the video features into the required embedding dimension
from torchvision import transforms
from torchvision.models.video import r3d_18




class KosmosTokenizer:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
        self.audio_tokenizer = Wav2Vec2Processor.from_pretrained("facebook/data2vec-audio-base-960h")
        #video
        self.tokenizer = T5Tokenizer.from_pretrained(
            "t5-large",
            additional_special_tokens=["<image>", "</image>", "<audio>", "</audio>", "<video>", "</video>"],
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

    def tokenize_texts(self, texts):
        texts =  self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids
        # Add image and audio tokens to text as "<s> <image> </image> <audio> </audio> text </s>"
        media_tokens = torch.tensor([[self.im_idx, self.im_end_idx, self.audio_idx, self.audio_end_idx, self.vid_idx, self.vid_end_idx]] * texts.shape[0])        
        return torch.cat([texts[:, 0:1], media_tokens, texts[:, 1:]], dim=1), texts

    def tokenize_images(self, images):
        return self.processor(images=images, return_tensors="pt").pixel_values
    
    def tokenize_audio(self, audios):
        return self.audio_tokenizer(audios, return_tensors="pt", padding=True, truncation=True).input_values
    
    def tokenize_videos(self, videos):
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
    def __init__(self):
        super().__init__()
        # Instantiate Clip Vit-l/14
        self.clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K").vision_model

        #audio model
        self.audio_model = Data2VecForCTC.from_pretrained("facebook/data2vec-audio-base-960h")

        #video
        self.video_model = r3d_18(pretrained=True)
        self.video_model = torch.nn.Sequential(*list(self.video_model.children())[:-1])




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

    def forward(self, text_tokens, images, audios, **kwargs):
        images = self.clip_model(pixel_values=images)["last_hidden_state"]
        images = self.perceive(images).squeeze(1)
        images = self.image_proj(images)

        # Process audio tokens
        audios = self.audio_model(audios).logits
        audios = audios.mean(dim=1)
        audios = self.audio_proj(audios)


        #process video tokens
        videos = videos.transpose(1, 2).contigous()
        videos = self.video_model(videos)
        videos = videos.view(videos.size(0), -1)
        videos = self.video_proj(videos)




        model_input = self.decoder.forward_embedding(text_tokens)[1]
        model_input = torch.cat([model_input[:, 0:6], images, audios, videos, model_input[:, 6:]], dim=1)
        model_input = self.decoder.forward_embedding(model_input, token_embedding=model_input)[0]

        return self.decoder(model_input, passed_x=model_input)[0]