import argparse
# import subprocess
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import spacy
from nltk import Tree
from itertools import permutations

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, help="Description of param1.")
parser.add_argument("--param2", type=str, help="Description of param2.")
parser.add_argument("--flag1", action="store_true", help="Enable flag1.")

args = parser.parse_args()


parsing = spacy.load("en_core_web_sm")
# load model
device = "cuda"

# pickscore
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pickscore = "yuvalkirstain/PickScore_v1"
processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pickscore).eval().to(device)

# prompt constituency parsing
def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

prompt = args.prompt
doc = parsing(prompt)

for sent in doc.sents:
    for np in sent.noun_chunks:
        print(np.text)

# print constituency parse tree
for sentence in doc.sents:
    tree = to_nltk_tree(sentence.root)
    tree.pretty_print()

# get 5 permutations
parsed_elements = [np.text for sent in doc.sents for np in sent.noun_chunks]
miniprompts_perm = list(permutations(parsed_elements))
miniprompts_perm = miniprompts_perm[:5]
print("5 permutations of generated miniprompts:")
for i in miniprompts_perm:
    print(i)

prompt_test = "A dog wearing a blue hat and sunglasses and a cat are riding a bicycle made out of spaghetti"


# fn to call train and inference scripts
def train_and_inference(exp_name, exp_dir, placeholder_token, super_category_token, prompts, seeds):
    # customize train.yaml config for specific token
    # only updating what need, and keep the rest as default
    train_config = f"""
        log:
          exp_name: {exp_name}
          exp_dir: {exp_dir}
          save_steps: 250
        data:
          train_data_dir: gen_images/{exp_name}
          placeholder_token: {placeholder_token}
          super_category_token: {super_category_token}
          dataloader_num_workers: 8
        model:
          pretrained_model_name_or_path: CompVis/stable-diffusion-v1-4
          use_nested_dropout: True
          nested_dropout_prob: 0.5
          normalize_mapper_output: True
          use_positional_encoding: True
          pe_sigmas: {'sigma_t': 0.03, 'sigma_l': 2.0}
          num_pe_time_anchors: 10
          output_bypass: True
        eval:
          validation_steps: 250
        optim:
          max_train_steps: 1000
          learning_rate: 1e-3
          train_batch_size: 2
          gradient_accumulation_steps: 4
            """

    train_config_path = 'input_configs/train.yaml'
    with open(train_config_path, 'w') as f:
        f.write(train_config)

    # run optimization script
    train_command = ['python', 'scripts/train.py', '--config_path', train_config_path]
    subprocess.run(train_command)

    # customize inference.yaml config for specific token
    # only updating what need, and keep the rest as default

    inference_config = f"""
        input_dir: {exp_dir}/{exp_name}
        prompts = {prompts}
        iteration: 250
        seeds: {seeds}
        torch_dtype: fp16
        truncation_idxs: [20, 250]
                """

    inference_config_path = 'input_configs/inference.yaml'
    with open(inference_config_path, 'w') as f:
        f.write(inference_config)

    # run inference script
    inference_command = ['python', 'scripts/inference.py', '--config_path', inference_config_path]
    subprocess.run(inference_command)


"""
exp_name = 'dog1'
exp_dir = 'experiments'
placeholder_token = 'N1'
super_category_token = 'dog'
prompts = ["A photo of a N1"]
seeds = [104, 8385, 24]
"""

train_and_inference(exp_name, exp_dir, placeholder_token, super_category_token, prompts, seeds)

for perm in miniprompts_perm:
    for mp in perm:
        train_and_inference(exp_name, exp_dir, placeholder_token, super_category_token, prompts, seeds)
