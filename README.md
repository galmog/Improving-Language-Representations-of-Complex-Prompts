## Improving the Language Representation of Prompts for Complex Text-to-Image Generation
### Project Overview

This project, submitted as the final project for my NLP course during my master's degree, aims to enhance diffusion-model based image generation from complex text prompts. The primary focus is on improving the language representation of embedded prompts to mitigate token neglect and incorrect attribute binding in complex image generation tasks.
### Motivation

Recent large-scale text-to-image models have given users the ability to intuitively generate new images from only textual prompts. However, current state of-the-art (SOTA) text-to-image models often fail to convey the semantics of long, complex prompts. The two most common issues are neglect and incorrect attribute binding; in neglect, the model completely fails to generate one or more subjects or attributes, and incorrect attribute binding refers to the model incorrectly binding a given attribute (ex. a colour) to the corresponding subject. This project seeks to address these challenges by enhancing the representation of language in text prompts, thus improving the overall quality and accuracy of the generated images.
### Approach
#### 1. Constituency-Based Parse Tree Construction
- **Method**: I use spaCy to construct a constituency-based parse tree from the input text prompt. This helps in breaking down complex prompts into more manageable components.
#### 2. Generation of Optimal Mini-Prompts
- **Method**: Based on the constituency parsing, we generate mini-prompts and optimal sequence for them. These mini-prompts are designed to improve the embedding accuracy and focus the cross-attention mechanism effectively.
#### 3. Iterative Image Generation
- **Method**: In the resulting optimal sequence, I apply a Textual Inversion technique on each mini-prompt; each mini-prompt is assigned a token, and I iteratively generate the complex image by nesting the subsequent tokens. This iterative approach helps in maintaining the integrity of the prompt's attributes and ensures better attribute binding in the final image.

This project is based on the original repository from the paper *[An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618)* by Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. The original repository can be found [here](https://github.com/rinongal/textual_inversion). I have built upon their work by adding the constituency-based parsing and mini-prompt generation approach to improve the overall performance of the model.
