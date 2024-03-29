[![Multi-Modality](images/agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Kosmos-X: Advanced Multi-Modality AI Model 🚀🌌

![Kosmos-X Next Generation Multi-Modality AI Model](images/kosmos-banner.png)

[![GitHub issues](https://img.shields.io/github/issues/kyegomez/Kosmos-X)](https://github.com/kyegomez/Kosmos-X/issues) 
[![GitHub forks](https://img.shields.io/github/forks/kyegomez/Kosmos-X)](https://github.com/kyegomez/Kosmos-X/network) 
[![GitHub stars](https://img.shields.io/github/stars/kyegomez/Kosmos-X)](https://github.com/kyegomez/Kosmos-X/stargazers) 
[![GitHub license](https://img.shields.io/github/license/kyegomez/Kosmos-X)](https://github.com/kyegomez/Kosmos-X/blob/main/LICENSE)

[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/Kosmos-X)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20AI%20project:%20Kosmos-X&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FKosmos-X) 
[![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FKosmos-X) 
[![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FKosmos-X&title=&summary=&source=)
![Discord](https://img.shields.io/discord/999382051935506503)
[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FKosmos-X&title=Kosmos-X%20-%20the%20next%20generation%20AI%20shields) 
[![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FKosmos-X&t=Kosmos-X%20-%20the%20next%20generation%20AI%20shields) 
[![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FKosmos-X&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=Kosmos-X%20-%20the%20next%20generation%20AI%20shields) 
[![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=Check%20out%20Kosmos-X%20-%20the%20next%20generation%20AI%20shields%20%23Kosmos-X%20%23AI%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2FKosmos-X)


## Installation
```bash
pip3 install --upgrade kosmosx
```

## Usage

```python
import torch
from kosmosx.model import Kosmos

# Create a sample text token tensor
text_tokens = torch.randint(0, 32002, (1, 50), dtype=torch.long)

# Create a sample image tensor
images = torch.randn(1, 3, 224, 224)

# Instantiate the model
model = Kosmos()

text_tokens = text_tokens.long()

# Pass the sample tensors to the model's forward function
output = model.forward(
    text_tokens=text_tokens,
    images=images
)

# Print the output from the model
print(f"Output: {output}")

```

# Training
Establish your configuration with: `accelerate config` then: `accelerate launch train.py`


## The model
KOSMOS-1 uses a decoder-only Transformer architecture based on [Magneto (Foundation Transformers)](https://arxiv.org/abs/2210.06423), i.e. an architecture that employs a so called sub-LN approach where layer normilization is added both before the attention module (pre-ln) and afterwards (post-ln) combining the advantages that either approaches have for language modelling and image understanding respectively. The model is also initialized according to a specific metric also described in the paper, allowing for more stable training at higher learning rates.

They encode images to image features using a CLIP VIT-L/14 model and use a [perceiver resampler](https://github.com/lucidrains/flamingo-pytorch) introduced in [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model) to pool the image features from `256 -> 64` tokens. The image features are combined with the token embeddings by adding them to the input sequence surrounded by special tokens `<image>` and `</image>`. An example is `<s> <image> image_features </image> text </s>`. This allows image(s) to be interwoven with text in the same sequence.

We follow the hyperparameters described in the paper visible in the following image:

![KOSMOS-1 Hyperparameters](./images/hyperparams.png)

## Details
### Model (decoder)
We use the torchscale implementation of the decoder-only Transformer architecture from Foundation Transformers:
    
```python
from torchscale.architecture.config import DecoderConfig
from torchscale.architecture.decoder import Decoder

config = DecoderConfig(
    decoder_layers=24,
    decoder_embed_dim=2048,
    decoder_ffn_embed_dim=8192,
    decoder_attention_heads=32,
    dropout=0.1,
    activation_fn="gelu",
    attention_dropout=0.1,
    vocab_size=32002,
    subln=True,                 # sub-LN approach
    xpos_rel_pos=True,          # rotary positional embeddings
    max_rel_pos=2048
)
decoder = Decoder(
    config,
    embed_tokens=embed,
    embed_positions=embed_positions,
    output_projection=output_projection
)
```


### CLIP VIT-L/14
For the image model (CLIP VIT-L/14) we use a pretrained OpenClip model:

```python
from transformers import CLIPModel
clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K").vision_model
# projects image to [batch_size, 256, 1024]
features = clip_model(pixel_values=images)["last_hidden_state"]
```

### Perceiver Resampler
We follow the default hyperparams for the perceiver resampler as no hyperparams are given in the paper:

```python
from flamingo_pytorch import PerceiverResampler
perceiver = PerceiverResampler(
    dim = 1024,
    depth = 2,
    dim_head = 64,
    heads = 8,
    num_latents = 64,
    num_media_embeds = 256
)
# projects image features to [batch_size, 64, 1024]
self.perceive(images).squeeze(1)
```

Because the model expects a hidden dimension of `2048`, we use a `nn.Linear` layer to project the image features to the correct dimension and initialize it according to Magneto's initialization scheme:

```python
image_proj = torch.nn.Linear(1024, 2048, bias=False)
torch.nn.init.normal_(
    image_proj.weight, mean=0, std=2048**-0.5
)
scaled_image_features = image_proj(image_features)
```

### Tokenizer
The paper describes a [SentencePiece](https://github.com/google/sentencepiece) with a vocabulary of `64007` tokens. For simplicity (as we don't have the training corpus available), we use the next best open-source alternative which is the pretrained [T5-large tokenizer](https://huggingface.co/t5-large) from HuggingFace. This tokenizer has a vocabulary of `32002` tokens.

```python
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained(
    "t5-large",
    additional_special_tokens=["<image>", "</image>"],
    extra_ids=0,
    model_max_length=1984 # 2048 - 64 (image features)
)
```
We then embed the tokens with a `nn.Embedding` layer. We actually use a `bnb.nn.Embedding` from
[bitandbytes](https://github.com/TimDettmers/bitsandbytes) which allows us to use 8-bit AdamW later.

```python
import bitsandbytes as bnb
embed = bnb.nn.Embedding(
    32002,          # Num embeddings
    2048,           # Embedding dim
    padding_idx
)
```

For positional embeddings, we use:
```python
from torchscale.component.embedding import PositionalEmbedding
embed_positions= PositionalEmbedding(
    2048,           # Num embeddings
    2048,           # Embedding dim
    padding_idx
)
```

Also, we add an output projection layer to project the hidden dimension to the vocabulary size and initialize it according to Magneto's initialization scheme:
```python
output_projection = torch.nn.Linear(
    2048, 32002, bias=False
)
torch.nn.init.normal_(
    output_projection.weight, mean=0, std=2048**-0.5
)
```

### Decoder changes
I had to make some slight changes to the decoder to allow it to accept already embedded features in the forward pass. This was necessary to allow the more complex input sequence described above. The changes are visible in the following diff in line 391 of `torchscale/architecture/decoder.py`:

```diff
+if kwargs.get("passed_x", None) is None:
+    x, _ = self.forward_embedding(
+        prev_output_tokens, token_embeddings, incremental_state
+    )
+else:
+    x = kwargs["passed_x"]

-x, _ = self.forward_embedding(
-    prev_output_tokens, token_embeddings, incremental_state
-)
```


---


### Dataset Strategy


Here is a markdown table with metadata for the datasets mentioned in the paper:

| Dataset | Description | Size | Link | 
|-|-|-|-|
| The Pile | Diverse English text corpus | 800 GB | [Huggingface](https://huggingface.co/datasets/the_pile) |
| Common Crawl | Web crawl data | - | [Common Crawl](https://commoncrawl.org/) |  
| LAION-400M | Image-text pairs from Common Crawl | 400M pairs | [Huggingface](https://huggingface.co/datasets/laion400m) |  
| LAION-2B | Image-text pairs from Common Crawl | 2B pairs | [ArXiv](https://arxiv.org/abs/2112.05251) |
| COYO | Image-text pairs from Common Crawl | 700M pairs | [Github](https://github.com/clovaai/coyo) |  
| Conceptual Captions | Image-alt text pairs | 15M pairs | [ArXiv](https://arxiv.org/abs/2103.01950) |
| Interleaved CC Data | Text and images from Common Crawl | 71M docs | Custom dataset |
| StoryCloze | Commonsense reasoning | 16k examples | [ACL Anthology](https://aclanthology.org/W17-0906/) |
| HellaSwag | Commonsense NLI | 70k examples | [ArXiv](https://arxiv.org/abs/1905.02875) |
| Winograd Schema | Word ambiguity | 273 examples | [PKRR 2012](https://doi.org/10.24963/kr.2012/26) |
| Winogrande | Word ambiguity | 1.7k examples | [AAAI 2020](https://arxiv.org/abs/1907.10641) |  
| PIQA | Physical commonsense QA | 16k examples | [AAAI 2020](https://arxiv.org/abs/1911.11641) |
| BoolQ | QA | 15k examples | [ACL 2019](https://aclanthology.org/N19-1246/) |
| CB | Natural language inference | 250 examples | [Sinn und Bedeutung 2019](https://semanticsarchive.net/Archive/DlZGNjZm/) | 
| COPA | Causal reasoning | 1k examples | [AAAI Spring Symposium 2011](https://www.aaai.org/ocs/index.php/SSS/SSS11/paper/download/2418/2874) |
| RelativeSize | Commonsense reasoning | 486 pairs | [ArXiv 2016](https://arxiv.org/abs/1602.00753) |
| MemoryColor | Commonsense reasoning | 720 examples | [ArXiv 2021](https://arxiv.org/abs/2109.11321) |
| ColorTerms | Commonsense reasoning | 320 examples | [ACL 2012](https://aclanthology.org/P12-2018/) |
| IQ Test | Nonverbal reasoning | 50 examples | Custom dataset |
| COCO Captions | Image captioning | 413k images | [PAMI 2015](https://doi.org/10.1109/TPAMI.2014.2366765) |  
| Flickr30k | Image captioning | 31k images | [TACL 2014](https://aclanthology.org/Q14-1010/) |
| VQAv2 | Visual QA | 1M QA pairs | [CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Goyal_Making_the_V_CVPR_2017_paper.pdf) |  
| VizWiz | Visual QA | 31k QA pairs | [CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gurari_VizWiz_Grand_Challenge_CVPR_2018_paper.pdf) |
| WebSRC | Web QA | 1.4k examples | [EMNLP 2021](https://aclanthology.org/2021.emnlp-main.261/) |  
| ImageNet | Image classification | 1.28M images | [CVPR 2009](https://doi.org/10.1109/CVPRW.2009.5206848) |
| CUB | Image classification | 200 bird species | [TOG 2011](https://vision.cornell.edu/se3/wp-content/uploads/2013/03/CUB_200_2011.pdf) |


----

## Todo
- [ ] Implement tokenizer for multi-modal processing
- [ ] Refactor training script
- [ ] Train 7B

# License
APACHE