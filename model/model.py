import torch
import time
import torch.nn as nn
import openai
import json
import requests
from transformers import BertModel, logging, AutoTokenizer, GPTJForCausalLM
logging.set_verbosity_error()


class gptj_model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        # GPT
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.gptj_dir)
        self.gpt = GPTJForCausalLM.from_pretrained(
            self.args.gptj_dir,
            revision='float16',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(self.args.device)

    def forward(self, ID, prompt_text: str) -> list[str]:
        input_ids = self.tokenizer(prompt_text, return_tensors='pt').to(self.args.device)
        input_len = input_ids.input_ids.shape[1]
        try:
            gen_tokens = self.gpt.generate(
                **input_ids,
                do_sample=True,
                temperature=0.1,
                max_length=input_len + 50,
                pad_token_id=self.tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"There has an ERROR in ID:{ID}")
            return ['[None]']
        gen_answer = self.tokenizer.batch_decode(gen_tokens[:, -50:])[0]  # string
        pred_triplet = gen_answer.strip().split('.\n')[0]
        pred_triplet_list = pred_triplet.strip().split('|')
        return pred_triplet_list  # list

class gpt_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        # generated from https://openai.com/
        openai.api_base = "https://********************************"
        openai.api_key = "sk-**************************************"
        self.args = args
        self.model_engine = "gpt-3.5-turbo"

    def forward(self, ID, prompt_text: str):
        prompt = [
            {"role":"user", "content": prompt_text}
        ]
        try:
            completion = openai.ChatCompletion.create(
                model=self.model_engine,
                messages=prompt,
                temperature=.0,
                max_tokens=50,
                stop='.'
            )
            if self.model_engine == "gpt-4":
                time.sleep(2)
        except Exception as e:
            print(f"\nThere has an ERROR in ID:{ID}")
            return ['[None]']
        pred_triplet = completion.choices[0].message.get('content')
        pred_triplet_list = pred_triplet.strip().split('|')
        return pred_triplet_list

class ernie_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        # generated from https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application
        self.API_KEY = "*************************"
        self.SECRET_KEY = "****************************"
        self.args = args

    def get_access_token(self):

        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": self.API_KEY, "client_secret": self.SECRET_KEY}
        return str(requests.post(url, params=params).json().get("access_token"))


    def forward(self, ID, prompt_text):
        # erbie-bot-ai-原生工作台
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ai_apaas?access_token=" + self.get_access_token()
        # llama model
        # url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_2_13b?access_token=" + self.get_access_token()
        try:
            payload = json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_text
                    }
                ],
                "temperature": 0.1
                # "top_p": 0.1
            })
            headers = {
                'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, data=payload)

            pred_triplet = json.loads(response.text)['result']
            pred_triplet_list = pred_triplet.strip().split('|')

            return pred_triplet_list
        except Exception as e:
            print(f"\nThere has an ERROR in ID:{ID}")
            return ['[None]']

