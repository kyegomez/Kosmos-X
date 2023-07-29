# Kosmos-X: Advanced Multi-Modality AI Model 🚀🌌

![Kosmos-X Next Generation Multi-Modality AI Model](/Kosmos-X-banner.png)

Kosmos-X is a groundbreaking modular Multi-Modality AI model designed to seamlessly process diverse forms of data, including:

* Images
* Videos
* Audio

The unique strength of Kosmos-X lies in its ability to process extremely long sequences of these multi-modality inputs, with context lengths of up to 40,000+!

## Model Roadmap
[Help us create a Model Roadmap on Kosmos-X Figma](https://www.figma.com/file/z3sNPzuB3thdOKT7oml6NI/Kosmos-X?type=whiteboard&node-id=1%3A142&t=Z37mybFxYALukurx-1)

## Ready for Training!

Kosmos-X is now ready for training, and we're actively seeking cloud providers or grant providers to collaborate in training this revolutionary model and eventually release it open source. If you're interested in learning more or supporting this endeavor, please feel free to email me at `kye@apac.ai`.

---

<div align="center">

[![GitHub issues](https://img.shields.io/github/issues/kyegomez/Kosmos-X)](https://github.com/kyegomez/Kosmos-X/issues) [![GitHub forks](https://img.shields.io/github/forks/kyegomez/Kosmos-X)](https://github.com/kyegomez/Kosmos-X/network) [![GitHub stars](https://img.shields.io/github/stars/kyegomez/Kosmos-X)](https://github.com/kyegomez/Kosmos-X/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/Kosmos-X)](https://github.com/kyegomez/Kosmos-X/blob/main/LICENSE)

[![Open Bounties](https://img.shields.io/endpoint?url=https%3A%2F%2Fconsole.algora.io%2Fapi%2Fshields%2Fkyegomez%2Fbounties%3Fstatus%3Dopen)](https://console.algora.io/org/kyegomez/bounties?status=open) [![Rewarded Bounties](https://img.shields.io/endpoint?url=https%3A%2F%2Fconsole.algora.io%2Fapi%2Fshields%2Fkyegomez%2Fbounties%3Fstatus%3Dcompleted)](https://console.algora.io/org/kyegomez/bounties?status=completed)

[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/Kosmos-X)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20AI%20project:%20Kosmos-X&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FKosmos-X) [![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FKosmos-X) [![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FKosmos-X&title=&summary=&source=)

![Discord](https://img.shields.io/discord/999382051935506503)


[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FKosmos-X&title=Kosmos-X%20-%20the%20next%20generation%20AI%20shields) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FKosmos-X&t=Kosmos-X%20-%20the%20next%20generation%20AI%20shields) [![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FKosmos-X&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=Kosmos-X%20-%20the%20next%20generation%20AI%20shields) [![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=Check%20out%20Kosmos-X%20-%20the%20next%20generation%20AI%20shields%20%23Kosmos-X%20%23AI%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2FKosmos-X)

</div>

---

Please note that this README.md has been recreated based on the provided information and may not fully reflect the actual content of the linked GitHub repository.

## Usage
This repo requires [apex](https://github.com/NVIDIA/apex#from-source) to be installed from source:
```bash

git clone https://github.com/kyegomez/Kosmos-X
cd Kosmos-X
# Basic requirements (transformers, torch, etc.)
pip install -r requirements.txt

# apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd ..

cd Kosmos
```

# Training

`accelerate config`

then: `accelerate launch train_distributed.py`

## Get Involved

We're just at the beginning of our journey. As we continue to develop and refine Kosmos-X, we invite you to join us. Whether you're a developer, researcher, or simply an enthusiast, your insights and contributions can help shape the future of Kosmos-X.

# Contributing to Kosmos-X

We are thrilled to invite you to be a part of the Kosmos-X project. This is not just an open source project but a community initiative, and we value your expertise and creativity. To show our appreciation, we have instituted a unique rewards system that directly compensates contributors from the revenue generated by the Kosmos-X API.

## Why Contribute

Contributing to Kosmos-X not only enhances your skills and profile but also comes with financial rewards. When you contribute code, documentation, or any form of improvement to the Kosmos-X project, you are adding value. As such, we believe it's only fair that you share in the rewards.

## Rewards Program

Here's how the Kosmos-X Rewards Program works:

1. **Submit a Pull Request:** This can be a code enhancement, bug fix, documentation update, new feature, or any improvement to the project.

2. **Review and Approval:** Our team will review your contribution. If it gets approved and merged, you become eligible for the rewards program.

3. **Revenue Share:** Once your pull request is merged, you will receive a percentage of the revenue generated by the Kosmos-X API. The percentage will be determined based on the significance and impact of your contribution. 

This means you're not just contributing to an open source project; you're becoming a part of the Kosmos-X ecosystem. Your efforts can yield ongoing benefits as the Kosmos-X API grows and evolves.

## Becoming a Paid API

As part of our growth strategy, we will be deploying Kosmos-X as a Paid API. The revenue generated from this API will not only sustain and further the project, but also fund the rewards program.

## How to Start Contributing

If you're ready to become a part of Kosmos-X and contribute to the future of multimodal embeddings, here's what you need to do:

1. Fork the repository.

2. Make your improvements or additions in your forked repository.

3. Submit a pull request detailing the changes you've made.

4. Our team will review your submission. If it's approved, it will be merged into the main repository, and you will become part of the Kosmos-X Rewards Program.

Thank you for considering contributing to Kosmos-X. Your expertise and commitment to this project are what make it thrive. Let's build the future of multimodal embeddings together.


## The model
KOSMOS-1 uses a decoder-only Transformer architecture based on [Magneto (Foundation Transformers)](https://arxiv.org/abs/2210.06423), i.e. an architecture that employs a so called sub-LN approach where layer normilization is added both before the attention module (pre-ln) and afterwards (post-ln) combining the advantages that either approaches have for language modelling and image understanding respectively. The model is also initialized according to a specific metric also described in the paper, allowing for more stable training at higher learning rates.

They encode images to image features using a CLIP VIT-L/14 model and use a [perceiver resampler](https://github.com/lucidrains/flamingo-pytorch) introduced in [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model) to pool the image features from `256 -> 64` tokens. The image features are combined with the token embeddings by adding them to the input sequence surrounded by special tokens `<image>` and `</image>`. An example is `<s> <image> image_features </image> text </s>`. This allows image(s) to be interwoven with text in the same sequence.

We follow the hyperparameters described in the paper visible in the following image:

![KOSMOS-1 Hyperparameters](./hyperparams.png)

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

### Training

* We're actively seeking cloud providers or grant providers to train this all-new revolutionary model and release it open source, if you would like to learn more please email me at kye@apac.ai


### TODO

* Integrate flash attention inside the `torchscale/component/multihead_attention.py`

* Integrate one write head is all you need

* Look into integrating qk_norm

* Look into integrating Falcon LLM model tokenizer if they allow special tokens

* Prepare datasets, training strategies, and infrastructure for massive production level traning

* Run tests and make sure trains well with all optimizations on small dataset

