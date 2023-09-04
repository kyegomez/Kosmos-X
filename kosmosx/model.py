import logging
import torch
import torch.nn as nn
from torch.nn import Module
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer

from flamingo_pytorch import PerceiverResampler

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Check if the modules are available
try:
    from torchscale.architecture.config import DecoderConfig
    from torchscale.architecture.decoder import Decoder
    from torchscale.component.embedding import PositionalEmbedding
    import bitsandbytes
except ImportError as e:
    logging.error(f"Failed to import module: {e}")
    raise

class KosmosTokenizer:
    """
    A tokenizer class for the kosmos model

    Attributes:
        processor(CLIPProcessor): The processor to tokenize images
        tokenizer: (AutoTokenizer): The tokenizer to tokenize text 
        im_idx: (int): The Index of the "<image>" token.
        im_end_idx (int): The index of the "</image>" token.
    """

    def __init__(self):
        try:
            self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b",
                additional_special_tokens=["<image>", "</image>"],
                eos_token="<eos>",
                pad_token="<pad>",
                extra_ids=0,
                model_max_length=8192
            )
        except Exception as e:
            logging.error(f"Failed to initialize KosmosTokenizer: {e}")
            raise

        self.im_idx, self.im_end_idx = self.tokenizer.convert_tokens_to_ids(["<image>", "</image>"])

    def tokenize_texts(self, texts: str):
        """
        Tokenize given texts.

        Args:
            Texts (str): The Text to be tokenized

        
        Returns:
            A tuple containing the tokenized texts and only the text tokens.
        """
        try:
            texts =  self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids
            # Add image tokens to text as "<s> <image> </image> text </s>"
            image_tokens = torch.tensor([[self.im_idx, self.im_end_idx]] * texts.shape[0])
            return torch.cat([texts[:, 0:1], image_tokens, texts[:, 1:]], dim=1), texts
        except Exception as e:
            logging.error(f"Failed to tokenize texts: {e}")
            raise

    def tokenize_images(self, images):
        """
        Tokenizes given images.

        Args:
            images: The images to be tokenized

        Returns:
            The tokenized images.
        
        """
        try:
            return self.processor(images=images, return_tensors="pt").pixel_values
        except Exception as e:
            logging.error(f"Failed to tokenize images: {e}")
            raise

    def tokenize(self, sample):
        """
        Tokenizes given sample.

        Args:
            Sample: The sample to be tokenized

        Returns:
            A dictionary containing the tokenized text tokens, images, labels, and attention mask.
        
        """
        try:
            text_tokens, only_text_tokens = self.tokenize_texts(sample["target_text"])
            attention_mask = text_tokens != self.tokenizer.pad_token_id
            dummy_image_features = torch.ones((text_tokens.shape[0], 64))
            attention_mask = torch.cat([dummy_image_features, attention_mask], dim=1)
            return {
                "text_tokens": text_tokens,
                "images": self.tokenize_images(sample["image"]),
                "labels": only_text_tokens,
                "attention_mask": attention_mask,
            }
        except Exception as e:
            logging.error(f"Failed to tokenize sample: {e}")
            raise

class Kosmos(nn.Module):
    """
    The main Kosmos model class.

    Attributes:
        clip_model (CLIPModel): The CLIP model for image processing.
        embed (Embedding): The embedding layer for tokens.
        embed_positions: (PositionEmbedding): The positional embedding layer.
        
        output_projection (Linear): the output projection layer.
        config (DecoderConfig): The configuration for the decoder
        decoder (Decoder): The decoder module

        perceieve(PerceiverResampler): The PerceieverResampler module for image processing.
        image_proj (Linear): The image projection layer.
    """

    def __init__(self):
        super().__init__()

        # Instantiate Clip Vit-l/14
        try:
            self.clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K").vision_model
        except Exception as e:
            logging.error(f"Failed to initialize CLIP model: {e}")
            raise

        self.embed = bitsandbytes.nn.modules.Embedding(32002, 2048, padding_idx=1)
        self.embed_positions= PositionalEmbedding(2048, 2048, 1)

        self.output_projection = nn.Linear(2048, 32002, bias=False)
        nn.init.normal_(self.output_projection.weight, mean=0, std=2048**-0.5)

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
            multiway=True,
            max_rel_pos=2048,
            alibi_pos_bias=True,
            alibi_num_heads=16,
        )
        
        try:
            self.decoder = Decoder(
                self.config,
                embed_tokens=self.embed,
                embed_positions=self.embed_positions,
                output_projection=self.output_projection
            )
        except Exception as e:
            logging.error(f"Failed to initialize Decoder: {e}")
            raise

        self.perceive = PerceiverResampler(
            dim = 1024,
            depth = 2,
            dim_head = 64,
            heads = 8,
            num_latents = 64,
            num_media_embeds = 257
        )

        self.image_proj = nn.Linear(1024, 2048, bias=False)
        nn.init.normal_(self.image_proj.weight, mean=0, std=2048**-0.5)

    def forward(self, text_tokens: torch.Tensor, images: torch.Tensor, **kwargs):
        """
       The forward pass for the Kosmos model.

       Args:
            text_tokens (torch.Tensor): The text tokens.
            images (torch.Tensor): The image tokens.

        Returns:
            The output of the decoder 
        
        """
        if not isinstance(text_tokens, torch.Tensor) or not isinstance(images, torch.Tensor):
            raise TypeError("text_tokens and images must be instances of torch.Tensor")

        try:
            images = self.clip_model(pixel_values=images)["last_hidden_state"]
            images = self.perceive(images).squeeze(1)
            images = self.image_proj(images)
        except Exception as e:
            logging.error(f"Failed during image processing: {e}")
            raise

        try:
            model_input = self.decoder.forward_embedding(text_tokens)[1]
            model_input = torch.cat([model_input[:, 0:2], images, model_input[:, 2:]], dim=1)
            model_input = self.decoder.forward_embedding(model_input, token_embedding=model_input)[0]
        except Exception as e:
            logging.error(f"Failed during text processing: {e}")
            raise

        try:
            return self.decoder(model_input, passed_x=model_input)[0]
        except Exception as e:
            logging.error(f"Failed during model forward pass: {e}")
            raise



class KosmosLanguage(Module):
    def __init__(self):
        super().__init__()
        self.embed = bitsandbytes.nn.modules.Embedding(
            320002,
            2048,
            padding_idx=1
        )

        self.embed_positions = PositionalEmbedding(
            2048,
            2048,
            1
        )

        self.output_projection = torch.nn.Linear(
            2048, 32002, bias=False
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
            multiway=True,
            max_rel_pos=2048,
            alibi_pos_bias=True,
            alibi_num_heads=16,
        )
        
        self.decoder = Decoder(
            self.config,
            embed_tokens=self.embed,
            embed_positions=self.embed_positions,
            output_projection=self.output_projection
        )


    def forward(self, text_tokens, **kwargs):
        model_input = self.decoder.forward_embedding(text_tokens)[0]
        return self.decoder(model_input, passed_x=model_input)[0]
    

