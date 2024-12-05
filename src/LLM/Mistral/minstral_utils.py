import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json

def load_quantized_model(model_name: str):
    """
    :param model_name: Name or path of the model to be loaded.
    :return: Loaded quantized model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                              # enable 4 bit quantization
        bnb_4bit_use_double_quant=True,                 # double quantization to optimize memory usage further
        bnb_4bit_quant_type="nf4",                      # uses normal float 4 for quantization
        bnb_4bit_compute_dtype=torch.bfloat16           # defines bfloat16 data type for intermediate computations
    )

    model = AutoModelForCausalLM.from_pretrained(       # load pretrained model weights
        model_name,
        # load_in_4bit=True,                             
        torch_dtype=torch.bfloat16,                     # Ensures all computations use bfloat16 precision.
        quantization_config=bnb_config                  # Applies the 4-bit quantization settings.
    )

    return model

def initialize_tokenizer(model_name: str):
    """
    Initialize the tokenizer with the specified model_name.

    :param model_name: Name or path of the model for tokenizer initialization.
    :return: Initialized tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1  # Set beginning of sentence token id
    return tokenizer

def generate_mistral_prompt_zero_shot(
        post_title, 
        post_content, 
        user_comment_history, 
        previous_comments_on_thread, 
        comment_of_interest,
    ):
    """
    Extracts zero_shot_json_to_skeleton json and create prompt to feed into minstral. 
    Returns minstral output in string format.
    """
    
    # Construct the Mistral prompt
    mistral_prompt = f"""
    [INST] 
    [Prompt]: "There is a particular user that we want to predict the response of on a Reddit post. Use the user's post, the post information, the comments in the thread, and the user's comment history to generate english response to the comment of interest."
    [The post's title]: {post_title}
    [The post's content]: {post_content}
    User's comment history]: {user_comment_history}
    [Previous user comments on the thread]: {previous_comments_on_thread}
    [Comment of interest]: {comment_of_interest}
    [/INST]
    """
    return mistral_prompt

def few_shot_json_to_skeleton(input_data):
    """
    Extracts zero_shot_json_to_skeleton json and create prompt to feed into minstral. 
    Returns minstral output in string format.
    """
    few_shot_prompt_response = {
        "example1": {
            "input": {
                "post_title": "",
                "post_content": "",
                "previous_comments_on_thread": "",
                "user_comment_history": ""
                },
                "comment_id": "string"
        },
        "example2": {
            "input": {
                "post_title": "",
                "post_content": "",
                "previous_comments_on_thread": "",
                "user_comment_history": ""
                },
                "comment_id": "string"
         },
        "example3": {
            "input": {
                "post_title": "",
                "post_content": "",
                "previous_comments_on_thread": "",
                "user_comment_history": ""
                },
                "comment_id": "string"
         }
    }  
    
    # extract relevant portion of input
    context = input_data['input']['context']
    comment_id = input_data['input']['comment_id']
    
    # Construct the Mistral prompt
    mistral_prompt = f"""
    [INST] 
    ["There is a particular user that we want to predict the response of on a Reddit post. Use the following information about this user of to generate an english response:]
    [The post's title]: {few_shot_prompt_response['example2']['input']['post_title']}
    [The post's content]: {few_shot_prompt_response['example2']['input']['post_content']}
    [Previous user comments on the thread]: {few_shot_prompt_response['example2']['input']['previous_comments_on_thread']}
    [User's comment history]: {few_shot_prompt_response['example2']['input']['user_comment_history']}

    ["There is a particular user that we want to predict the response of on a Reddit post. Use the following information about this user of to generate an english response:]
    [The post's title]: {few_shot_prompt_response['example2']['input']['post_title']}
    [The post's content]: {few_shot_prompt_response['example2']['input']['post_content']}
    [Previous user comments on the thread]: {few_shot_prompt_response['example2']['input']['previous_comments_on_thread']}
    [User's comment history]: {few_shot_prompt_response['example2']['input']['user_comment_history']}

    ["There is a particular user that we want to predict the response of on a Reddit post. Use the following information about this user of to generate an english response:]
    [The post's title]: {few_shot_prompt_response['example3']['input']['post_title']}
    [The post's content]: {few_shot_prompt_response['example3']['input']['post_content']}
    [Previous user comments on the thread]: {few_shot_prompt_response['example3']['input']['previous_comments_on_thread']}
    [User's comment history]: {few_shot_prompt_response['example3']['input']['user_comment_history']}

    ["There is a particular user that we want to predict the response of on a Reddit post. Use the following information about this user of to generate an english response:]
    [The post's title]: {context['post_title']}
    [The post's content]: {context['post_content']}
    [Previous user comments on the thread]: {context['previous_comments_on_thread']}
    [User's comment history]: {context['user_comment_history']}

    [/INST]
    """
    
    return mistral_prompt
