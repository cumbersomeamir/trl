#!pip3 install torch peft tqdm transformers

from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
form trl.core import LengthSampler

#Define the parse arguments
@dataclass
class ScriptArguments:
  """
  The name of the Causal LM we wish to fine-tine with PPO
  """

  #Note: gpt2 uses conv1d instead of linear layers which are not yet supported in 8 bit mode
  # model like gpt-neo are more suitable
  model_name: Optional[str] = field(default="edbeeching/gpt-neo-1.3B-imdb", metadata={"help": "The model name"})
  log_with: Optional[str] = field(default = None , metadata={"help": "use wandb to login"})
  learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning date"})
  merge_model_adapter: Optional[bool] = field(default=False, metadata={"help": "merging using model adapter"})

parser = HFArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


config = PPOConfig(
                   model_name = script_args.model_name,
                   learning_rate = script_args.learning_rate,
                   log_with = script_args.log_with,
                   batch_size = 64,
                   mini_batch_size = 4,
                   optmize_cuda_cache=True
    
                   )

#We then define the arguments to pass through the sentiment analysis pipeline
# We set the return_all_score to True to get the sentiment score for each token

sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": config.mini_batch_size}


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.

def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
  #Returns : dataloader (torch.utils.data.DataLoader) - The dataloader for the dataset
  tokenizer = AutoTokenizer.from_pretrained(config.model_name)
  tokenizer.pad_token = tokenizer.eos_token

  #Load IMDB with datasets
  ds = load_dataset(dataset_name, split="train")
  ds = ds.rename_columns({"text": "review"})
  ds = df.filter(lambda x: len(x["review"])>200, batched=False)

  imput_size = LengthSampler(input_min_text_length, input_max_text_length)

  def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["input_ids"])[: input_size()]
    sample["query"] = tokenizer.encode(sample["review"])
    return sample

  ds = ds.map(tokenize, batched= False)
  ds.set_format(type="torch")
  return ds

  dataset =build_dataset(config)

  def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

    #set seed before initilising the value head for deterministic eval
    set_seed(config.seed)


"Apply LORA"

def print_trainable_parameters(model):
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()

    print(f"trainable params:{trainable_params} || all params: {all_param} || trainable%: {100 *trainable_params / all_params}")


lora_config = LoraConfig(
    r =16,
    lora_alpha=32,
    lora_dropout = 0.05,
    bias ="none",
    task_type = "CAUSAL_LM"
    
)



model = AutoModelForCausalLMWithValueHead.from_pretrained(
    
    config.model_name,
    load_in_8bit = True,
    device_map="balanced",
    max_memory={0:"800MB", 1: "800MB"},
    peft_config = lora_config
)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)

print_trainable_parameters(model)

tokenizer.pad_token = tokenizer.eos_token
optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr = config.learning_rate)

# We then build the PPOTrainer, passing the model, the reference model and tokenizer
ppo_trainer = PPOTrainer(
    config, model, ref_model = None, tokenizer = tokenizer, dataset = dataset, data_collator= collator, optimizer = optmizer
)

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
  device = 0 if torch.cuda.is_available() else "cpu"

  sentiment_pipe = pipeline("sentiment-analysis", model= "lvwerra/distilbert-imdb", device = device)

  # We then define the arguments to define into the sentiment function
  # These argurments are passed to the generate functino of the PPO Trainer, which is wrapped aorun the generate fuction of the trained model

  generation_kwargs = {
      "min_length" : -1,
      "top_k" : 0.0,
      "top_p" :1.0.
      "do_sample" : True,
      "pad_token_id": tokenizer.eos_token_id,
      "eos_token_id": -1
  }
d
output_min_length = 4
output_max_length = 16

output_length_sampler = LengthSampler(output_min_length,output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
  query_tensors = batch["input_ids"]

  model.gradient_checkpointing_disable()
  model.pretrained_model.config.use_cache = True

  #Get Responses for Causal LM
  response_tensors = []
  for query in query_tensors:
    gen_len = output_length_sampler()
    generation_kwards["max_new_tokens"] = gen_len
    response = ppo_trainer.generate(query, **generation_kwards)
    response_tensors.append(response.squeeze()[-gen_len:])
  batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors] 

  #Compute the sentiment score
  text = [q + r for q, r in zip(batch[query], batch["response"])] #What does the zip function do
  pipe_outputs = sentiment_pipe(texts, **sent_kwargs) #What does these two ** do?
  rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

  #Run PPO step
  model.gradient_checkpointing_enable()
  model.pretrained_model.config.use_cache = False

  stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
  ppo_trainer.log_stats(stats, batch, rewards)

model.push_to_hub(f"{script_args.model_name}-ppo-sentiment")
