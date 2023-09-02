from transformers import CLIPModel
import argparse
import torch

def get_placeholder_loop(placeholder_string, embedder):
    new_placeholder = None

    while True:
        if new_placeholder is None:
            new_placeholder = input(
                f"Placeholder string {placeholder_string} was already used. Please enter a replacement string: ")
        else:
            new_placeholder = input(
                f"Placeholder string '{new_placeholder}' maps to more than a single token. Please enter another string: ")

        token = get_clip_token_for_string(embedder.tokenizer, new_placeholder)
        if token is not None:
            return new_placeholder, token

def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]

    if torch.count_nonzero(tokens - 49407) == 2:
        return tokens[0, 1]

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manager_ckpts",
        type=str,
        nargs="+",
        required=True,
        help="Paths to a set of embedding managers to be merged."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the merged manager",
    )

    args = parser.parse_args()

    # Initialize the CLIP model (replace with your desired model and configuration)
    embedder = CLIPModel.from_pretrained("openai/clip-vit-base").cuda()

    string_to_token_dict = {}
    string_to_param_dict = torch.nn.ParameterDict()

    placeholder_to_src = {}

    for manager_ckpt in args.manager_ckpts:
        print(f"Parsing {manager_ckpt}...")

        # Load embeddings directly here (replace with your actual loading code)
        # You may need to modify this part to load the embeddings correctly
        embeddings = {}  # Load embeddings from manager_ckpt

        for placeholder_string, token in embeddings.items():
            if placeholder_string not in string_to_token_dict:
                string_to_token_dict[placeholder_string] = token

                # Create a dummy parameter (you may need to replace this with actual parameters)
                dummy_param = torch.nn.Parameter(torch.Tensor([0]))
                string_to_param_dict[placeholder_string] = dummy_param

                placeholder_to_src[placeholder_string] = manager_ckpt
            else:
                new_placeholder, new_token = get_placeholder_loop(placeholder_string, embedder)
                string_to_token_dict[new_placeholder] = new_token

                # Create a dummy parameter (you may need to replace this with actual parameters)
                dummy_param = torch.nn.Parameter(torch.Tensor([0]))
                string_to_param_dict[new_placeholder] = dummy_param

                placeholder_to_src[new_placeholder] = manager_ckpt

    print("Saving combined manager...")
    # Save string_to_token_dict and string_to_param_dict directly
    # You may need to modify this part to save the embeddings correctly
    combined_embeddings = {
        "string_to_token_dict": string_to_token_dict,
        "string_to_param_dict": string_to_param_dict
    }

    # Save combined_embeddings to args.output_path

    print("Managers merged. Final list of placeholders: ")
    print(placeholder_to_src)
