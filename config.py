import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:

  @dataclass
  class OpenAI:
    API_KEY = os.environ.get('OPENAI_API_KEY')
