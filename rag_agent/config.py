from dotenv import load_dotenv
import os

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
BRAVE_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")  # required for web search
