import torch
from torchscale.architecture.config import DecoderConfig
from torchscale.architecture.decoder import Decoder
from torchscale.component.embedding import PositionalEmbedding
from transformers import T5Tokenizer, CLIPProcessor, CLIPModel, PreTrainedTokenizerFast
from tokenizers import SentencePieceBPETokenizer
from transformers import Wav2Vec2Tokenizer
from transformers import Wav2Vec2Model

from flamingo_pytorch import PerceiverResampler
from PIL import Image
from torch.nn import Embedding, Module
import bitsandbytes


class KosmosTokenizer:
    def __init__(self, modalities=["text", "image", "audio"]):
        self.modalities = modalities
        self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")

        # T5 uses SentencePiece tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(
            "t5-large",
            additional_special_tokens=["<image>", "</image>", "<audio>", "</audio>"],
            extra_ids=0,
            model_max_length=1984
        )
        self.audio_idx, self.audio_end_idx = self.tokenizer.convert_tokens_to_ids(["<audio>", "</audio>"])
        self.im_idx, self.im_end_idx = self.tokenizer.convert_tokens_to_ids(["<image>", "</image>"])
        self.audio_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")


    def tokenize_texts(self, texts):
        texts =  self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids
        # Add image and audio tokens to text as "<s> <image> </image> <audio> </audio> text </s>"
        media_tokens = torch.tensor([[self.im_idx, self.im_end_idx, self.audio_idx, self.audio_end_idx]] * texts.shape[0])
        return torch.cat([texts[:, 0:1], media_tokens, texts[:, 1:]], dim=1), texts

    def tokenize_images(self, images):
        return self.processor(images=images, return_tensors="pt").pixel_values
    
    def tokenize_audio(self, audios):
        return self.audio_tokenizer(audios, return_tensors="pt", padding=True, truncation=True).input_ids

    def tokenize(self, target_texts):
      text_tokens_list, only_text_tokens_list = [], []
      max_length = 0

      for target_text in target_texts:
          text_tokens, only_text_tokens = self.tokenize_texts(target_text)
          text_tokens_list.append(text_tokens)
          only_text_tokens_list.append(only_text_tokens)
          max_length = max(max_length, text_tokens.shape[1])

      padded_text_tokens_list = []
      padded_only_text_tokens_list = []

      for text_tokens, only_text_tokens in zip(text_tokens_list, only_text_tokens_list):
          padded_text_tokens = torch.cat([text_tokens, torch.full((1, max_length - text_tokens.shape[1]), self.tokenizer.pad_token_id, dtype=torch.long)], dim=1)
          padded_only_text_tokens = torch.cat([only_text_tokens, torch.full((1, max_length - only_text_tokens.shape[1]), self.tokenizer.pad_token_id, dtype=torch.long)], dim=1)
          padded_text_tokens_list.append(padded_text_tokens)
          padded_only_text_tokens_list.append(padded_only_text_tokens)

      attention_mask = torch.stack(padded_text_tokens_list) != self.tokenizer.pad_token_id

      return {
          "text_tokens": torch.stack(padded_text_tokens_list),
          "labels": torch.stack(padded_only_text_tokens_list),
          "attention_mask": attention_mask,
      }

class Kosmos(Module):
    def __init__(self, modalities=["text", "image", "audio"]):
        super().__init__()
        # Instantiate Clip Vit-l/14
        self.modalities = modalities
        self.clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K").vision_model
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

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

        #add audio
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_proj = torch.nn.Linear(768, 2048, bias=False)
        torch.nn.init.normal_(
            self.audio_proj.weight, mean=0, std=2048 ** -0.5
        )

    def forward(self, text_tokens, images=None, audios=None, **kwargs):
        if "image" in self.modalities and images is not None:
            images = self.clip_model(pixel_values=images)["last_hidden_state"]
            images = self.perceive(images).squeeze(1)
            images = self.image_proj(images)

        if "audio" in self.modalities and audios is not None:
            audios = self.audio_model(input_ids=audios).last_hidden_state
            audios = audios.mean(dim=1)
            audios = self.audio_proj(audios)

        model_input = self.decoder.forward_embedding(text_tokens)[1]
        if "image" in self.modalities and images is not None and "audio" in self.modalities and audios is not None:
            model_input = torch.cat([model_input[:, 0:3], images, audios, model_input[:, 3:]], dim=1)
        elif "image" in self.modalities and images is not None:
            model_input = torch.cat([model_input[:, 0:3], images, model_input[:, 3:]], dim=1)
        elif "audio" in self.modalities and audios is not None:
            model_input = torch.cat([model_input[:, 0:3], audios, model_input[:, 3:]], dim=1)

        model_input = self.decoder.forward_embedding(model_input, token_embedding=model_input)[0]

        return self.decoder(model_input, passed_x=model_input)[0]

import time

import torch
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import get_scheduler, default_data_collator, get_linear_schedule_with_warmup
from torch.optim import AdamW

# from kosmos import Kosmos, KosmosTokenizer
from accelerate import Accelerator

from rich.progress import Progress
from datasets import Image
from bitsandbytes.optim import AdamW8bit
from lion_pytorch import Lion


from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist

AWS_ACCESS_KEY_ID= 'AKIA5K4H36GT5EVDX2MA'
AWS_SECRET_ACCESS_KEY= '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'
WANDB_API_KEY= '0fc08bb0e90314a2bb602afa0b2e6cf56abc3f49'

#logging
import boto3


#training
import wandb

from torch.utils.tensorboard import SummaryWriter



def save_model_to_s3(model, bucket_name, key_prefix, step):
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    model_path = f"checkpoint_at_step_{step}.pt"
    torch.save(model.state_dict(), model_path)
    s3.upload_file(model_path, bucket_name, f"{key_prefix}/{model_path}")



def count_number_of_parameters(model, only_trainable: bool = True) -> int:
    if only_trainable:
        num_params: int = sum(p.numel()
                              for p in model.parameters() if p.requires_grad)
    else:
        num_params: int = sum(p.numel() for p in model.parameters() if p)
    return int(num_params)


# def load_alpaca_cot_dataset(data_dir: str) -> DatasetDict:
#     data_dir = Path(data_dir)
#     dataset = {"train": [], "validation": []}

#     for split in dataset.keys():
#         for file in (data_dir / split).glob("*json"):
#             with open(file, "r") as f:
#                 data = json.load(f)
#                 dataset[split].extend(data)
    
#     return DatasetDict({split: Dataset.from_dict({"data": data}) for split, data in dataset.items()})

def prep_sample(sample):
    instruction = sample["instruction"]
    input_text = sample["input"]
    output_text = sample["output"]
    text = f"Instruction: {instruction} Input: {input_text} Output: {output_text}"
    return {
        "target_text": text
    }

def train(args):

    if args.use_ddp:
        dist.init_process_group(backend="nccl")


    accelerator = Accelerator(
        mixed_precision="fp16"
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    #v1
    model = Kosmos()
    # if args.use_ddp:
        # model = DistributedDataParallel(model)
    # else:
        # model = DataParallel(model)

    model = model.to(accelerator.device)

    #device count
    if torch.cuda.device_count() > 1:
        print(f"Let's use ${torch.cuda.device_count()} GPUS")




    optimizer = Lion(model.parameters(), lr=args.learning_rate / 3, weight_decay=args.weight_decay * 3)
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    tokenizer = KosmosTokenizer(modalities=["text"])


    # dataset = load_dataset("QingyiSi/Alpaca-CoT", split="train[:1%]")
    # dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1%]")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")


    # dataset = dataset.map(prep_sample, num_proc=8)
    dataset = dataset.map(prep_sample, num_proc=8)

    # dataset = dataset.map(lambda sample: tokenizer(sample["target_text"]), batched=True, batch_size=128, remove_columns=["instruction", "input", "output"])
    dataset = dataset.map(lambda sample: (print(sample), tokenizer.tokenize(sample))[1], batched=True, batch_size=128, remove_columns=["instruction", "input", "output"], input_columns=["target_text"])
    train_dataloader = DataLoader(
        dataset, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True
    )

    #====================> load data #====================> load data #====================> load data #====================> load data 

    model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(model, train_dataloader, optimizer,
                                                                           lr_scheduler)
    model.train()
    accelerator.register_for_checkpointing(lr_scheduler)

    model.clip_model.requires_grad_(False)
    model.clip_model.encoder.layers[-1].requires_grad_(True)

    accelerator.print(
        f"Number of parameters: {count_number_of_parameters(model):,}")
    accelerator.print(
        f"Number of trainable parameters: {count_number_of_parameters(model, only_trainable=True):,}")

    # Log model and optimizer parameters to wandb
    accelerator.init_trackers(project_name="kosmos")

    #wandb
    wandb.init(project="kosmos", config=args)
    
    #init tensorboard writer
    tb_writer = SummaryWriter()



    train_loader = iter(train_dataloader)
    epoch_loss = 0
    total_loss = 0
    start_time = time.time()

    with Progress() as progress:
        task = progress.add_task("[red]Training...", total=args.max_steps)
        for step in range(0, args.max_steps):
            batch_start = time.time()
            batch = {key: value for key, value in next(train_loader).items() if key != "images"}
            outputs = model(**batch, self_attn_padding_mask=batch["attention_mask"])
            # Shift so that tokens < n predict n
            outputs = torch.cat([outputs[:, :1], outputs[:, 67:]], dim=1).contiguous()
            # shift_logits = outputs[..., :-1, :].contiguous()
            # shift_labels = batch["labels"][..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            one_hot_labels = torch.nn.functional.one_hot(batch["labels"][:, 1:], num_classes=32002).float()
            loss = loss_fct(outputs[:,:-1], one_hot_labels)

            epoch_loss += loss.detach().float()

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            batch_end = time.time()
            logs = {
                "loss": loss.item(),
                "perplexity": torch.exp(loss).item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "examples": args.batch_size * (step + 1),
                "examples_per_second": args.batch_size / (batch_end - batch_start),
            }
            if step % args.log_every == args.log_every - 1:
                #log metrics to wandb
                wandb.log(logs, step=step)

                #log metrics to tensorboard 
                                # Log metrics to TensorBoard
                tb_writer.add_scalar("loss", logs["loss"], step)
                tb_writer.add_scalar("perplexity", logs["perplexity"], step)
                tb_writer.add_scalar("lr", logs["lr"], step)
                tb_writer.add_scalar("examples", logs["examples"], step)
                tb_writer.add_scalar("examples_per_second", logs["examples_per_second"], step)

                #accelerator
                accelerator.log(logs, step=step)
                progress.update(task, advance=1, description=f"Step Loss: {loss.item():.5f} "
                                                             f"| Mean Loss: {(total_loss + epoch_loss) / step:.5f} "
                                                             f"| Mean PPL: {torch.exp((total_loss + epoch_loss) / step):.2f} "
                                                             f"| Examples: {args.batch_size * (step + 1)} "
                                                             f"| Examples/s: {args.batch_size / (batch_end - batch_start):.2f} "
                                                             f"| Elapsed: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")

            if step % args.save_every == args.save_every - 1:
                train_epoch_loss = epoch_loss / args.save_every
                total_loss += epoch_loss
                epoch_loss = 0

                accelerator.log({
                    "train_ppl": torch.exp(train_epoch_loss),
                    "train_epoch_loss": train_epoch_loss,
                }, step=step)

                progress.print(f"Saving checkpoint at step {step}...")
                accelerator.save_state(
                    f"{args.checkpoint_dir}/checkpoint_at_step_{step}/")
                
                #save the model weights to s3 
                save_model_to_s3(model, "kosmostraining", "kosmosv1/checkpoints", step)
                print(f"Saved to s3: {save_model_to_s3} ")

        #finish tensorboard writer
        tb_writer.close()

        #finish wnabd run
        wandb.finish()


class Args:
    def __init__(self):
        self.checkpoint_dir = "checkpoints"
        self.learning_rate = 1e-5
        self.weight_decay = 0.01
        self.warmup_steps = 0
        self.max_steps = 100000
        self.batch_size = 4
        self.log_every = 1
        self.save_every = 100
        self.seed = None
        self.use_ddp = False

args = Args()
train(args)