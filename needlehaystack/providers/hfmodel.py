import os
from operator import itemgetter
from typing import Optional

import torch
import tensor_parallel as tp

from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain_openai import ChatOpenAI  
from langchain.prompts import PromptTemplate
import tiktoken
import asyncio

from vllm import LLM, SamplingParams

from .model import ModelProvider

# for llama
def reset_llama_rope(model, model_max_train_len, scaling_factor):
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len, device="cpu", dtype=torch.float32)
    return

class HFmodel(ModelProvider):
    """
    A wrapper class for interacting with OpenAI's API, providing methods to encode text, generate prompts,
    evaluate models, and create LangChain runnables for language model interactions.

    Attributes:
        model_name (str): The name of the OpenAI model to use for evaluations and interactions.
        model (AsyncOpenAI): An instance of the AsyncOpenAI client for asynchronous API calls.
        tokenizer: A tokenizer instance for encoding and decoding text to and from token representations.
    """
        
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens=131072, max_new_tokens=200, do_sample=True, num_beams=4, min_new_tokens=30)

    def __init__(self,
                 model_name: str = "microsoft/Phi-3-mini-128k-instruct",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        """
        Initializes the OpenAI model provider with a specific model.

        Args:
            model_name (str): The name of the OpenAI model to use. Defaults to 'gpt-3.5-turbo-0125'.
            model_kwargs (dict): Model configuration. Defaults to {max_tokens: 300, temperature: 0}.
        
        Raises:
            ValueError: If NIAH_MODEL_API_KEY is not found in the environment.
        """
        api_key = os.getenv('NIAH_MODEL_API_KEY')
        if (not api_key):
            raise ValueError("NIAH_MODEL_API_KEY must be in env.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.api_key = api_key
        
        self.model = LLM(model=model_name, dtype=torch.float16, max_model_len=131072, enforce_eager=True, tensor_parallel_size=8, trust_remote_code=True, disable_custom_all_reduce=True)
        
        # scaling_factor = 10 # hardcode
        # we should change this line, `model_max_train_len` will be modified in the future as a param.
        # reset_rope(self.model, model_max_train_len=81920, scaling_factor=scaling_factor)
        
        # self.model = tp.tensor_parallel(self.model, sharded=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", trust_remote_code=True)
    
    def evaluate_model(self, prompt: str) -> str:
        """
        Evaluates a given prompt using the HF model and retrieves the model's response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The content of the model's response to the prompt.
        """
        # prompt = self.encode_text_to_tokens(prompt)
        # input_ids = prompt['input_ids'].to(self.model.device)
        # response_ids = self.model.generate(input_ids, **self.model_kwargs)
        
        # response = self.decode_tokens(
        #     tokens=response_ids[0],
        #     context_length=50,
        #     start_id=input_ids.shape[1]
        # ).strip()

        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, min_tokens=30, max_tokens=131072, early_stopping=False)
        response = self.model.generate(prompt, sampling_params)

        return response[0].outputs[0].text.strip()
    
    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        """
        Generates a structured prompt for querying the model, based on a given context and retrieval question.

        Args:
            context (str): The context or background information relevant to the question.
            retrieval_question (str): The specific question to be answered by the model.

        Returns:
            list[dict[str, str]]: A list of dictionaries representing the structured prompt, including roles and content for system and user messages.
        """
        return [f"You are a helpful AI bot that answers questions for a user. Keep your response short and direct.\nBased on this context: {context} and Answer the question:\n{retrieval_question} Don't give information outside the document or repeat your findings.\n"]

        # return [f"This is a very long story book: {context}\nBased on the content of the book, {retrieval_question}\nAnswer:"]
            
    
    def encode_text_to_tokens(self, text: str) -> list[int]:
        """
        Encodes a given text string to a sequence of tokens using the model's tokenizer.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        return self.tokenizer.encode(text)
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        """
        Decodes a sequence of tokens back into a text string using the model's tokenizer.

        Args:
            tokens (list[int]): The sequence of token IDs to decode.
            context_length (Optional[int], optional): An optional length specifying the number of tokens to decode. If not provided, decodes all tokens.

        Returns:
            str: The decoded text string.
        """
        return self.tokenizer.decode(tokens[:context_length])
    
    def get_langchain_runnable(self, context: str) -> str:
        """
        Creates a LangChain runnable that constructs a prompt based on a given context and a question, 
        queries the OpenAI model, and returns the model's response. This method leverages the LangChain 
        library to build a sequence of operations: extracting input variables, generating a prompt, 
        querying the model, and processing the response.

        Args:
            context (str): The context or background information relevant to the user's question. 
            This context is provided to the model to aid in generating relevant and accurate responses.

        Returns:
            str: A LangChain runnable object that can be executed to obtain the model's response to a 
            dynamically provided question. The runnable encapsulates the entire process from prompt 
            generation to response retrieval.

        Example:
            To use the runnable:
                - Define the context and question.
                - Execute the runnable with these parameters to get the model's response.
        """

        pass
    

