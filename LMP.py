
import openai
from time import sleep
import time
import os
import pickle
import json
import hashlib
from termcolor import cprint
try:
    from vlpart.demo.demo import run_segment
except:
    from demo.demo import run_segment
import base64
import requests
import cv2

with open('key.txt', 'r') as file:
    KEY = file.read().rstrip()
openai.api_key = KEY


class GPT4V:

    def __init__(self, api_key):
        self.api_key = api_key

    # Function to encode the image
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_response(self, image_path):
        # Getting the base64 string
        base64_image = self.encode_image(image_path)
        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
        "model": "gpt-4o",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "List the name of the objects on the table with simplest words. such as: cup, bottle, book. Please only answer objects name. Do not say anything else."
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"

                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response = response.json()['choices'][0]['message']['content']

        return response


class LMP:
    def __init__(self, image_path, is_offline = False, object_name = None, part_name = None):
        self._stop_tokens = ['# Query: ']
        self._cache = DiskCache(load_cache=True)
        self.image_path = image_path
        self.is_offline = is_offline
        self.object_name = object_name
        self.part_name = part_name

    def build_prompt(self, query):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(curr_dir, 'prompt.txt')
        with open(full_path, 'r') as f:
            contents = f.read().strip()
        prompt = contents + '\n Query: ' + query
        return prompt

    def _cached_api_call(self, **kwargs):
        # add special prompt for chat endpoint
        user1 = kwargs.pop('prompt')
        new_query = '# Query:' + user1.split('# Query:')[-1]
        user1 = ''.join(user1.split('# Query:')[:-1]).strip()
        assistant1 = f'Got it. I will complete what you give me next.'
        user2 = new_query
        messages=[
            {"role": "system", "content": "You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a robot arm in a tabletop environment."},
            {"role": "user", "content": user1},
            {"role": "assistant", "content": assistant1},
            {"role": "user", "content": user2},
        ]
        kwargs['messages'] = messages
        if kwargs in self._cache:
            print('(using cache)', end=' ')
            return self._cache[kwargs]
        else:
            ret = openai.chat.completions.create(**kwargs).choices[0].message.content
            cprint(f"code generated from gpt: \n{ret}", "yellow")
            self._cache[kwargs] = ret
            return ret

    def __call__(self, query, **kwargs):
        prompt = self.build_prompt(query)

        start_time = time.time()
        if self.is_offline:
            code_str = f"run_segment('{self.object_name} {self.part_name}', image_path)\n# done"
        else: 
            while True:
                try:
                    code_str = self._cached_api_call(
                        prompt=prompt,
                        stop=self._stop_tokens,
                        temperature=0,
                        model='gpt-4',
                        max_tokens=256
                    )
                    break
                except:
                    print('call api error')
                    print('Retrying after 3s.')
                    sleep(3)
            print(f'*** OpenAI API call took {time.time() - start_time:.2f}s ***')

        to_exec = code_str
        
        gvars = {'run_segment': run_segment}
        lvars = {'image_path': self.image_path}
        self.exec_code(to_exec, gvars, lvars)


    def exec_code(self, code_str, gvars=None, lvars=None):
        banned_phrases = ['import', '__']
        for phrase in banned_phrases:
            assert phrase not in code_str
        try:
            exec(code_str, gvars, lvars)
        except Exception as e:
            print(f'Error executing code:\n{code_str}')
            raise e

class DiskCache:
    """
    A convenient disk cache that stores key-value pairs on disk.
    Useful for querying LLM API.
    """
    def __init__(self, cache_dir='cache', load_cache=True):
        self.cache_dir = cache_dir
        self.data = {}

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        else:
            if load_cache:
                self._load_cache()

    def _generate_filename(self, key):
        key_str = json.dumps(key)
        key_hash = hashlib.sha1(key_str.encode('utf-8')).hexdigest()
        return f"{key_hash}.pkl"

    def _load_cache(self):
        for filename in os.listdir(self.cache_dir):
            with open(os.path.join(self.cache_dir, filename), 'rb') as file:
                key, value = pickle.load(file)
                self.data[json.dumps(key)] = value

    def _save_to_disk(self, key, value):
        filename = self._generate_filename(key)
        with open(os.path.join(self.cache_dir, filename), 'wb') as file:
            pickle.dump((key, value), file)

    def __setitem__(self, key, value):
        str_key = json.dumps(key)
        self.data[str_key] = value
        self._save_to_disk(key, value)

    def __getitem__(self, key):
        str_key = json.dumps(key)
        return self.data[str_key]

    def __contains__(self, key):
        str_key = json.dumps(key)
        return str_key in self.data

    def __repr__(self):
        return repr(self.data)

def run_lmp(image_path= "datasets/clutter_sample/color_0001.png"):
    #------------
    # affordance = 'hold' #'hand over'
    affordance = 'pour'
    # image_path = "/workspaces/inference_container/exp_images/color_test2.png"
    file_name = os.path.basename(image_path)
    # New file name
    new_file_name = "cropped_" + file_name

    # Get the directory part of the original path
    directory = os.path.dirname(image_path)

    # Construct the new path by joining the directory with the new file name
    cropped_image_path = os.path.join(directory, new_file_name)
    # cropped_image_path = "/workspaces/inference_container/exp_images/cropped_color_test2.png"

    def crop_image(image_path):
        image = cv2.imread(image_path)
        new_image = image[200:630, 530:930] # normal object
        # new_image = image[200:720, 430:1030] # bigger object

        cv2.imwrite(cropped_image_path, new_image)
    crop_image(image_path)
    #-----------

    API_key = KEY
    mllm = GPT4V(api_key=API_key)
    object = mllm.get_response(cropped_image_path)
    print(object)
    # object = 'red bowl'
    lmp = LMP(cropped_image_path)
    query = 'affordance is ' + affordance + ' and ' + 'object is ' + object
    lmp(query)

if __name__ == "__main__":
    run_lmp()