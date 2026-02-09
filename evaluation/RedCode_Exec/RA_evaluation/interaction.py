import json
import logging
import os
import signal
import time

import docker
import torch
# from openai import AzureOpenAI, OpenAI
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer

from .sysprompt import (SAFETY_SYS_SUFFIX0, SAFETY_SYS_SUFFIX1,
                       SAFETY_SYS_SUFFIX2, SAFETY_SYS_SUFFIX3,
                       SAFETY_SYS_SUFFIX4)
try:
    import replicate
except Exception:
    replicate = None



class BaseModel:
    def __init__(self, model, temperature, top_p, seed, max_tokens, dry_run):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_tokens = max_tokens
        self.dry_run = dry_run

        self.use_api = False  # <--- 新增：是否通过 OpenAI兼容API
        self.use_replicate = False
        self.replicate_model = None

        # --- Universal Replicate override (推荐) ---
        if os.getenv("REPLICATE_API_TOKEN") and os.getenv("REPLICATE_MODEL"):
            if replicate is None:
                raise RuntimeError("replicate 库未安装。请先 `pip install replicate`")
            self.use_replicate = True
            self.replicate_model = os.getenv("REPLICATE_MODEL")


        if model == 'deepseek-coder-6.7b-instruct':
            ds_base = os.getenv('DEEPSEEK_API_BASE')
            ds_key  = os.getenv('DEEPSEEK_API_KEY')
            if ds_base and ds_key:
                openai.api_base = ds_base
                openai.api_key  = ds_key

                self.use_api = True
                self.model = os.getenv("DEEPSEEK_API_MODEL", "deepseek-chat")
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                local_path = os.getenv(
                    "DEEPSEEK_LOCAL_PATH",
                    "/root/local_model/deepseek-ai/deepseek-coder-6.7b-instruct/"
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.bfloat16 if device == "cuda" else torch.float32

                self.tokenizer = AutoTokenizer.from_pretrained(
                    local_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
                self.deepseek = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    local_files_only=True
                ).to(device)


        if model in ("gpt-4o-2024-05-13", "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-5.1"):
            openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise RuntimeError("缺少 OPENAI_API_KEY 环境变量")
            self.use_api = True
            self.model = model

        if model=='deepseek-coder-v2-lite-instruct':
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-v2-lite-instruct", trust_remote_code=True)
            self.deepseek_v2 = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/deepseek-coder-v2-lite-instruct",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        if model=='meta-llama-3-8B-instruct':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/meta-llama-3-8B-instruct")
            self.llama3 = AutoModelForCausalLM.from_pretrained(
                "meta-llama/meta-llama-3-8B-instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        if model=='meta-llama-3.1-8b-instruct':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/meta-llama-3.1-8b-instruct")
            self.llama3 = AutoModelForCausalLM.from_pretrained(
                "meta-llama/meta-llama-3.1-8b-instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        if model=='meta-llama-3-70b-instruct':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/meta-llama-3-70b-instruct")
            self.llama3 = AutoModelForCausalLM.from_pretrained(
                "meta-llama/meta-llama-3-70b-instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        if model=='meta-llama-3.1-70b-instruct':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/meta-llama-3.1-70b-instruct")
            self.llama3 = AutoModelForCausalLM.from_pretrained(
                "meta-llama/meta-llama-3.1-70b-instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        if model=='llama-2-7b-chat-hf':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/llama-2-7b-chat-hf")
            self.llama2 = AutoModelForCausalLM.from_pretrained(
                "meta-llama/llama-2-7b-chat-hf",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        if model=='llama-2-13b-chat-hf':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/llama-2-13b-chat-hf")
            self.llama2 = AutoModelForCausalLM.from_pretrained(
                "meta-llama/llama-2-13b-chat-hf",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        if model=='llama-2-70b-chat-hf':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/llama-2-70b-chat-hf")
            self.llama2 = AutoModelForCausalLM.from_pretrained(
                "meta-llama/llama-2-70b-chat-hf",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        if model=='gemma-2-9b-it':
            self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
            self.gemma2 = AutoModelForCausalLM.from_pretrained(
                "google/gemma-2-9b-it",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        if model=='codeqwen1.5-7b-chat':
            model_id = "qwen/codeqwen1.5-7b-chat"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.codeqwen = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto"
            )

    def generate_deepseek(self, messages):
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.deepseek.device)
        # tokenizer.eos_token_id is the id of <|EOT|> token
        outputs = self.deepseek.generate(inputs, max_new_tokens=self.max_tokens, top_p=self.top_p, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
        ans = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return ans
    def generate_deepseek_v2(self, messages):
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.deepseek_v2.device)
        # tokenizer.eos_token_id is the id of <|EOT|> token
        outputs = self.deepseek_v2.generate(inputs, max_new_tokens=self.max_tokens, do_sample=False, top_p=self.top_p, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
        ans = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return ans


    def _messages_to_prompt(self, messages):
        system_lines = []
        dialog_lines = []
        for m in messages:
            role = (m.get("role") or "").strip().lower()
            content = m.get("content") or ""
            if role == "system":
                system_lines.append(content)
            elif role in ("user", "assistant"):
                dialog_lines.append(f"{role.upper()}: {content}")
            else:
                # 兜底：未知角色也当作普通文本
                dialog_lines.append(f"{role.upper() or 'USER'}: {content}")
        sys_part = ""
        if system_lines:
            sys_part = "SYSTEM:\n" + "\n".join(system_lines) + "\n\n"
        return sys_part + "\n".join(dialog_lines) + "\nASSISTANT:"

    def generate_replicate(self, messages):
        assert self.use_replicate and self.replicate_model, "Replicate 未初始化"

        prompt = self._messages_to_prompt(messages).strip()

        if len(prompt) > 120000:
            prompt = prompt[-120000:]

        replicate_input = {
            "prompt": prompt,
        }

        try:
            out = replicate.run(
                self.replicate_model,
                input=replicate_input,
            )
        except Exception as e:
            try:
                out = replicate.run(
                    self.replicate_model,
                    input={
                        **replicate_input,
                        "temperature": float(self.temperature),
                        "top_p": float(self.top_p),
                        "max_tokens": int(self.max_tokens),
                    },
                )
            except Exception as e2:
                raise RuntimeError(
                    f"Replicate 调用失败：第一次(仅prompt)异常={e}; 第二次(带生成参数)异常={e2}; "
                    f"model={self.replicate_model!r}"
                )

        if hasattr(out, "__iter__") and not isinstance(out, (str, bytes)):
            return "".join(map(str, out))
        return str(out)

    def generate_openai(self, messages, max_retries=10, backoff_factor=1):
        attempt = 0
        while attempt < max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_completion_tokens=self.max_tokens,
                    messages=messages
                )
                return response["choices"][0]["message"]["content"]
            except Exception:
                attempt += 1
                time.sleep(backoff_factor * (2 ** attempt))
        raise Exception(f"Failed to get a response from OpenAI API after {max_retries} attempts.")


    def generate_llama3(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.llama3.device)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.llama3.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            eos_token_id=terminators,
            do_sample=False,
            top_p=self.top_p,
        )
        ans = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return ans
    def generate_llama2(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.llama2.device)
        outputs = self.llama2.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            do_sample=False,
            top_p=self.top_p,
        )
        ans = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return ans
    def generate_gemma2(self, messages):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(self.gemma2.device)
        outputs = self.gemma2.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            do_sample=False,
            top_p=self.top_p,
        )
        ans = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return ans
    def generate_codeqwen(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.codeqwen.device)
        generated_ids = self.codeqwen.generate(
            model_inputs.input_ids,
            max_new_tokens=self.max_tokens,
            do_sample=False,
            top_p=self.top_p,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    def generate_claude(self, messages, max_retries=10, backoff_factor=1):
        attempt = 0
        system_prompt = ""
        new_messages = []
        for message in messages:
            if message['role']=='system':
                system_prompt = message['content']
            else:
                new_messages.append(message)

        if not system_prompt:
            while attempt < max_retries:
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        messages=new_messages
                    )

                    # Accessing the response content (text)
                    return response['choices'][0]['message']['content']
                except Exception as e:
                    attempt += 1
                    wait_time = backoff_factor * (2 ** attempt)
                    print(f"Error: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        else:
            while attempt < max_retries:
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        messages=new_messages
                    )

                    # Accessing the response content (text)
                    return response['choices'][0]['message']['content']

                except Exception as e:
                    attempt += 1
                    wait_time = backoff_factor * (2 ** attempt)
                    print(f"Error: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

        raise Exception(f"Failed to get a response from the Claude API after {max_retries} attempts.")
        # except Exception as e:


class EvalTask:
    def __init__(self, task, version, safety_sys_choice):
        self.task = task
        self.version = version

        # Options: ['none', '0_safety_sys', '1_safety_sys_awareness', '2_safety_sys_detection', '3_safety_sys_modification']
        if safety_sys_choice=='none':
            self.safety_sys = ""
        elif safety_sys_choice=='0_safety_sys':
            self.safety_sys = SAFETY_SYS_SUFFIX0
        elif safety_sys_choice=='1_safety_sys_awareness':
            self.safety_sys = SAFETY_SYS_SUFFIX1
        elif safety_sys_choice=='2_safety_sys_detection':
            self.safety_sys = SAFETY_SYS_SUFFIX2
        elif safety_sys_choice=='3_safety_sys_modification':
            self.safety_sys = SAFETY_SYS_SUFFIX3
        elif safety_sys_choice=='4_safety_integration':
            self.safety_sys = SAFETY_SYS_SUFFIX4
        else:
            raise ValueError(f"Unsupported safety system prompt: {safety_sys_choice}")

class Container:
    def __init__(self, image, name=None):
        self.image = image
        self.name = name or image
        self.client = docker.from_env()

        # Check if the image exists, and build it if not
        if not self.image_exists(self.image):
            logging.info(f"Image {self.image} not found. Building the image from Dockerfile.")
            self.build_image()

        # Create the container
        self.container = self.client.containers.run(
            self.image,
            labels={"created_by": "code-agent-evaluation"},
            detach=True, tty=True, stdin_open=True,
            name=self.name
        )
        logging.info(f"Container {self.name} started successfully.")

        # Run initialization script
        output = self.container.exec_run("bash /app/init.sh")
        output_str = output.output.decode()
        logging.info(f"Container init output: {output_str}")

    # def __del__(self):
    #     try:
    #         self.container.stop()
    #     except:
    #         pass
    #     finally:
    #         self.container.remove()

    def __del__(self):
        try:
            if hasattr(self, "container") and self.container:
                try:
                    self.container.stop()
                except:
                    pass
                try:
                    self.container.remove()
                except:
                    pass
        except:
            pass

    def __enter__(self):

        try:
            logging.info(f"Starting container {self.name} in __enter__...")
            output = self.container.exec_run("bash /app/init.sh")
            output_str = output.output.decode()
            logging.info(f"Container init output: {output_str}")
        except Exception as e:
            logging.error(f"Failed to start container: {e}")
            self.container = None
        return self

    def image_exists(self, image_name):
        try:
            # Attempt to fetch the image
            self.client.images.get(image_name)
            return True
        except docker.errors.ImageNotFound:
            return False

    def build_image(self):
        try:
            dockerfile_path = os.path.join(os.path.dirname(__file__), "../../../environment")
            logging.info(f"Building image {self.image} from Dockerfile in {dockerfile_path}.")
            self.client.images.build(path=dockerfile_path, tag=self.image)
            logging.info(f"Image {self.image} built successfully.")
        except Exception as e:
            logging.error(f"Failed to build image {self.image}: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container.stop()
        self.container.remove()

    def execute_with_retries(self, cmd, retries=3, timeout=1*60):
        def handler(signum, frame):
            raise TimeoutError("Execution timed out")

        signal.signal(signal.SIGALRM, handler)

        for attempt in range(retries):

            # # Naive exec_run
            # output = self.container.exec_run(cmd, stdin=False, stdout=True, stderr=True, demux=False)
            # output_str = output.output.decode()
            # return output_str

            try:
                logging.debug(f"Attempt {attempt+1}: Starting exec_run in {cmd}")
                signal.alarm(timeout)
                output = self.container.exec_run(cmd, stdin=False, stdout=True, stderr=True, demux=False)
                signal.alarm(0)
                output_str = output.output.decode()
                logging.debug("Finished exec_run")

                if "EOFError" in output_str:
                    logging.error(f"EOFError detected in command output: {cmd}\nOutput:\n{output_str}")
                    return "Execution blocked due to EOFError"

                return output_str
            except TimeoutError:
                logging.warning(f"Attempt {attempt+1}: Command execution timed out in docker container with cmd:\n```{cmd}```")
            except Exception as e:
                logging.error(f"Attempt {attempt+1}: An error occurred in docker container: {e}")

            logging.info(f"Retrying in 2 seconds...")
            time.sleep(2)

        return "Execution timed out"

    def execute_independent(self, command, timeout=1*60, *params):
        # print("=== EXECUTING INDEPENDENT ===\n", command)
        language, command = command
        # if params:
        #     print("== Parameters ==\n", params)
        if language == "bash":
            cmd = ["bash", "-c", command]
            if params:
                cmd.append("--")
                cmd.extend(params)
        elif language == "python":
            cmd = ["python", "-c", command, *params]
        elif language == "c++":
            self.execute_independent(
                (
                    "bash",
                    f'echo "{json.dumps(command)}" > /tmp/main.cpp && '
                    f"g++ -o /tmp/a.out /tmp/main.cpp",
                ),
                None,
            )
            cmd = ["/tmp/a.out", *params]
        elif language == "c":
            self.execute_independent(
                (
                    "bash",
                    f'echo "{json.dumps(command)}" > /tmp/main.cpp && '
                    f"gcc -o /tmp/a.out /tmp/main.cpp",
                ),
                None,
            )
            cmd = ["/tmp/a.out", *params]
        else:
            raise ValueError(f"Unsupported language {language}")

        return self.execute_with_retries(cmd, timeout=timeout)