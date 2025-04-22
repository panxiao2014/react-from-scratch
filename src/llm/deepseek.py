from openai import OpenAI
from src.config.logging import logger

class DeepSeek:
    def __init__(self) -> None:
        #read txt file ./credentials/deepseek.txt and get api key:
        with open("./credentials/deepseek.txt", "r") as f:
            api_key = f.read()
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )

            if not response.choices[0].message.content:
                logger.error("Empty response from the DeepSeek")
                return None
            
            cache_hit_tokens = getattr(response.usage, 'prompt_cache_hit_tokens', 0)
            cache_miss_tokens = getattr(response.usage, 'prompt_cache_miss_tokens', 0)         
            logger.info(f"Generated response. Cache hit: {cache_hit_tokens}, miss: {cache_miss_tokens}")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"DeekSeek error generating response: {e}")
            return None