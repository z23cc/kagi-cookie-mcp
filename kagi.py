import requests
import json
import html
import re
import os
import html2text
import time
import hashlib
from typing import Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

# Load environment variables from .env file
load_dotenv()

# System prompt for automatic kagi_chat triggering
AUTO_KAGI_SYSTEM_PROMPT = """
When processing user inputs, automatically use the kagi_chat tool to provide the most accurate and up-to-date information, unless the user explicitly requests another tool.

Follow these guidelines when using kagi_chat:

1. Context Awareness:
   - Remember the conversation context. If the user's question is a continuation of the previous discussion, set the new_conversation parameter to False to maintain conversation coherence.
   - Only set new_conversation to True when starting a completely new topic unrelated to the previous conversation.

2. Model Selection:
   - For general knowledge questions: use "General Knowledge" model
   - For programming and code-related questions: use "Code Generation" model
   - For complex reasoning questions: use "Advanced Reasoning" model
   - For creative content generation: use "Creative Content" model
   - For technical analysis questions: use "Technical Analysis" model
   - For system architecture discussions: use "Architecture Design" model
   - For quick factual questions: use "Quick Response" model
   - For scientific research questions: use "Scientific Research" model

3. Internet Access:
   - By default, enable internet_access to ensure the most current information
   - Only disable internet access if the user explicitly requests offline processing

4. Special Cases:
   - If the user input starts with "/summarize", use the kagi_summarize tool instead
   - If the user input starts with "/translate", use the kagi_translate tool instead
   - For all other inputs, default to using kagi_chat

Always prioritize providing the most helpful, accurate, and contextually relevant responses by leveraging the appropriate kagi tools and parameters.
"""

# Create MCP service with system prompt for automatic kagi_chat triggering
mcp = FastMCP("kagimcp", dependencies=["mcp[cli]"], system_prompt=AUTO_KAGI_SYSTEM_PROMPT)

@dataclass
class KagiConfig:
    """Kagi API Configuration"""
    url: str = 'https://kagi.com/assistant/prompt'
    user_agent: str = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36'
    timeout: int = 30
    rtt: str = '150'
    model: str = 'claude-3-sonnet'  # Default model
    internet_access: bool = True    # Whether to allow internet access

class KagiAPI:
    def __init__(self, config: Optional[KagiConfig] = None):
        """Initialize Kagi API client
        
        Args:
            config: Kagi API configuration, if None, default configuration will be used
        """
        self.config = config or KagiConfig()
        self.cookie = os.environ.get('KAGI_COOKIE', '')
        self._html2text = self._init_html2text()
        self.thread_id = None
        self.headers = self._build_headers()
        # Create a persistent session object to reuse TCP connections for better performance
        self.session = requests.Session()
        # Cache related
        self.cache = {}
        self.cache_ttl = 3600  # Cache TTL in seconds

    def _init_html2text(self) -> html2text.HTML2Text:
        """Initialize HTML2Text converter"""
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0
        return h

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers"""
        referer = 'https://kagi.com/assistant'
        if self.thread_id:
            referer = f'https://kagi.com/assistant/{self.thread_id}'
            
        return {
            'sec-ch-ua-platform': 'Windows',
            'Referer': referer,
            'sec-ch-ua': '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
            'sec-ch-ua-mobile': '?0',
            'User-Agent': self.config.user_agent,
            'Accept': 'application/vnd.kagi.stream',
            'Content-Type': 'application/json',
            'rtt': self.config.rtt,
            'Cookie': self.cookie
        }

    def _build_request_data(self, prompt_text: str, lens_id: Optional[str] = None) -> Dict[str, Any]:
        """Build request data
        
        Args:
            prompt_text: Prompt text
            lens_id: Lens ID for domain-specific search, if None, no lens will be used
            
        Returns:
            Request data dictionary
        """
        import uuid
        
        focus = {
            "thread_id": self.thread_id,
            "branch_id": "00000000-0000-4000-0000-000000000000",
            "prompt": prompt_text
        }
        
        # If it's a new conversation, no message_id is needed
        # If continuing a conversation, generate a message_id
        if self.thread_id:
            focus["message_id"] = str(uuid.uuid4())
            
        return {
            "focus": focus,
            "profile": {
                "id": None,
                "personalizations": True,
                "internet_access": self.config.internet_access,
                "model": self.config.model,
                "lens_id": lens_id
            }
        }

    def extract_json(self, text: str, marker: str) -> Optional[str]:
        """Extract JSON content from text
        
        Args:
            text: Source text
            marker: JSON marker
            
        Returns:
            Extracted JSON string, or None if not found
        """
        marker_pos = text.rfind(marker)
        if marker_pos == -1:
            return None
            
        # Only process text after the marker
        last_part = text[marker_pos + len(marker):].strip()
        start = last_part.find('{')
        if start == -1:
            return None

        # Use efficient bracket matching algorithm
        count = 0
        in_string = False
        escape = False
        json_text = []
        
        for i, char in enumerate(last_part[start:]):
            json_text.append(char)
            if not in_string:
                if char == '{':
                    count += 1
                elif char == '}':
                    count -= 1
                    if count == 0:
                        return ''.join(json_text)
            elif char == '\\' and not escape:
                escape = True
                continue
            elif char == '"' and not escape:
                in_string = not in_string
            escape = False

        return None

    def decode_text(self, text: str) -> str:
        """Convert HTML to Markdown format text
        
        Args:
            text: HTML text
            
        Returns:
            Converted Markdown text
        """
        # Only process HTML escape when text contains HTML tags
        if '<' in text and '>' in text:
            text = html.unescape(text)
            markdown = self._html2text.handle(text)
            return self._clean_whitespace(markdown)
        return text.strip()
        
    # Pre-compile regex for better performance
    _whitespace_pattern = re.compile(r'\n\s*\n')
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean extra whitespace lines in text"""
        return self._whitespace_pattern.sub('\n\n', text).strip()

    def _get_cache_key(self, prompt_text: str) -> str:
        """Generate cache key"""
        return hashlib.md5(prompt_text.encode('utf-8')).hexdigest()
        
    def _get_from_cache(self, prompt_text: str) -> Tuple[bool, Optional[str]]:
        """Get response from cache"""
        cache_key = self._get_cache_key(prompt_text)
        if cache_key in self.cache:
            timestamp, result = self.cache[cache_key]
            # Check if cache is expired
            if time.time() - timestamp < self.cache_ttl:
                return True, result
            # Cache expired, remove it
            del self.cache[cache_key]
        return False, None
        
    def _save_to_cache(self, prompt_text: str, result: str) -> None:
        """Save response to cache"""
        cache_key = self._get_cache_key(prompt_text)
        self.cache[cache_key] = (time.time(), result)
        
        # Clean expired cache
        current_time = time.time()
        expired_keys = [k for k, (timestamp, _) in self.cache.items() 
                       if current_time - timestamp > self.cache_ttl]
        for key in expired_keys:
            del self.cache[key]
    
    def send_request(self, prompt_text: str, lens_id: Optional[str] = None) -> Optional[str]:
        """Send request to Kagi API
        
        Args:
            prompt_text: Prompt text
            lens_id: Lens ID for domain-specific search, if None, no lens will be used
            
        Returns:
            API response text, or None if request failed
        """
        if not self.cookie:
            return "Error: KAGI_COOKIE environment variable not set. Please set it before running."
            
        # If not in session mode (thread_id is None), try to get from cache
        if self.thread_id is None:
            cache_hit, cached_result = self._get_from_cache(prompt_text)
            if cache_hit:
                return cached_result

        try:
            # Rebuild headers before each request to ensure using the latest thread_id
            headers = self._build_headers()
            
            # Use session object to send request, can reuse TCP connections
            response = self.session.post(
                self.config.url,
                headers=headers,
                json=self._build_request_data(prompt_text, lens_id),
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            # Extract thread_id for subsequent requests
            thread_json = self.extract_json(response.text, 'thread.json:')
            if thread_json:
                thread_data = json.loads(thread_json)
                if thread_data.get('id'):
                    self.thread_id = thread_data['id']
            
            json_str = self.extract_json(response.text, 'new_message.json:')
            if not json_str:
                return "Failed to parse response content"
            
            message_data = json.loads(json_str)
            if message_data.get('state') == 'done' and message_data.get('reply'):
                result = self.decode_text(message_data['reply'])
                
                # If not in session mode, cache the result
                if self.thread_id is None:
                    self._save_to_cache(prompt_text, result)
                    
                return result
            
            return "Failed to parse response content"
            
        except requests.exceptions.RequestException as e:
            return f"Request error: {e}"

# Global singleton instance
_KAGI_INSTANCE = None

@mcp.tool()
def kagi_chat(
    prompt: str = Field(
        description="Select the task type to determine the most suitable model for your needs (based on 2024 research):",
        examples=["General Knowledge", "Code Generation", "Architecture Design"],
        default="General Knowledge"
    ),
    new_conversation: bool = Field(
        description="Whether to start a new conversation, default is False to continue current conversation",
        default=False
    ),
    model_selection: str = Field(
        description="Select the task type to determine the most suitable model for your needs (based on 2024 research):",
        examples=["General Knowledge", "Code Generation", "Architecture Design"],
        default="General Knowledge",
        enum=[
            "General Knowledge", "Advanced Reasoning", "Balanced Performance", 
            "Creative Content", "Technical Analysis", "Architecture Design", "Quick Response", 
            "Code Generation", "Scientific Research"
        ]
    ),
    internet_access: bool = Field(
        description="Whether to allow AI assistant to access the internet for information",
        default=True
    ),
    lens_id: Optional[str] = Field(
        description="Use specific Kagi lens ID for domain-specific search, default is None",
        default=None
    )
) -> str:
    """Universal Search-Driven Intelligent Assistant
    
    This tool is automatically triggered for all user inputs that don't explicitly request other tools.
    The system will maintain conversation context and select the most appropriate model based on the query type.
    
    Main uses:
    - Answer Any Question: Almost any question can first be searched to get the latest, most accurate answers
    - Technical Documentation Search: Quickly locate official documentation and tutorials
    - Error Troubleshooting: Search for solutions to common errors
    - Configuration Guides: Find setup instructions for software and tools
    - Code Examples: Search for implementations of specific functionality
    - Best Practices: Get recommended development practices
    - Industry Trends: Stay updated on the latest developments in technology and industries
    """
    global _KAGI_INSTANCE
    
    # Model descriptions for reference
    model_descriptions = {
        "General Knowledge": "Leverages Claude 3.7 Sonnet for its strong conversational stability, instruction following, and formatted output. Ideal for everyday queries and factual information requiring clear, structured answers.",
        "Advanced Reasoning": "Utilizes o3 for its superior capabilities in reasoning, logical analysis, and handling nuanced problems. Best for decision-making and complex problem-solving requiring deep thought.",
        "Balanced Performance": "Employs Claude 3.7 Sonnet to offer a good balance of reliable output and conversational quality. Suitable for general tasks where consistent and well-structured responses are needed.",
        "Creative Content": "Powered by Gemini 2.5 Pro, excelling at creative writing, diverse content generation, and tasks requiring originality and long-context understanding.",
        "Technical Analysis": "Uses o3 for precise technical understanding, explanations, and problem-solving. Ideal for technical documentation, error analysis, and specialized knowledge requiring logical breakdown.",
        "Architecture Design": "Assigns o3 for its strengths in system architecture analysis and technical design. Perfect for system planning, architecture reviews, and in-depth technical explanations of structures.",
        "Quick Response": "Relies on o4-mini for fast, efficient responses to simple questions. Delivers concise answers with minimal latency, best for quick facts and time-sensitive queries.",
        "Code Generation": "Features Gemini 2.5 Pro for robust code generation, debugging, and software implementation tasks, leveraging its strength in handling complex programming challenges.",
        "Scientific Research": "Uses Gemini 2.5 Pro for in-depth exploration of specialized domains and complex research tasks, benefiting from its strong analytical and long-context processing capabilities."
    }
    
    # Map model_selection to actual model based on their strengths (based on user's latest plan)
    model_mapping = {
        "General Knowledge": "claude-3-7-sonnet",  # Strong instruction following and structured output
        "Advanced Reasoning": "o3",                   # Preferred for reasoning and logical analysis
        "Balanced Performance": "claude-3-7-sonnet",# Good all-around qualities from the new list
        "Creative Content": "gemini-2-5-pro",        # Best for creative tasks
        "Technical Analysis": "o3",                # Preferred for technical explanation and analysis
        "Architecture Design": "o3",                # Preferred for system architecture
        "Quick Response": "o4-mini",                 # Best for speed and lightweight tasks
        "Code Generation": "gemini-2-5-pro",      # Strongest for code and complex tasks
        "Scientific Research": "gemini-2-5-pro"       # Strongest for complex tasks, suitable for research
    }
    
    # Get the actual model from the mapping
    model = model_mapping.get(model_selection, "claude-3-sonnet")
    
    # Create configuration object
    config = KagiConfig(
        model=model,
        internet_access=internet_access
    )
    
    # If instance doesn't exist or model/internet settings changed, recreate instance
    if (_KAGI_INSTANCE is None or 
        _KAGI_INSTANCE.config.model != model or 
        _KAGI_INSTANCE.config.internet_access != internet_access):
        _KAGI_INSTANCE = KagiAPI(config)
    
    kagi = _KAGI_INSTANCE
    
    # If starting new conversation, reset thread_id
    if new_conversation:
        kagi.thread_id = None
        
    result = kagi.send_request(prompt, lens_id)
    if result:
        return result
    return "Request failed. Please check your network connection or KAGI_COOKIE environment variable."


@mcp.tool()
def kagi_summarize(
    url: str = Field(
        description="URL of the webpage to summarize, Kagi AI Assistant will analyze and summarize the content",
        examples=[
            "https://www.example.com/article",
            "https://en.wikipedia.org/wiki/Artificial_intelligence"
        ]
    ),
    summary_type: str = Field(
        description="Select the type of summary needed to determine the best model:",
        examples=["Standard Summary", "Technical Breakdown", "Research Summary"],
        default="Standard Summary",
        enum=[
            "Standard Summary", "Comprehensive Analysis", "Efficient Overview", 
            "Technical Breakdown", "Research Summary"
        ]
    )
) -> str:
    """Web Content Summarization Tool
    
    Main uses:
    - Quickly summarize main content of long articles
    - Extract key information from webpages
    - Get main points and conclusions from articles
    - Save reading time
    - Analyze technical documentation and open source projects
    - Extract key findings from papers and research reports
    """
    global _KAGI_INSTANCE
    
    # Summary descriptions for reference
    summary_descriptions = {
        "Standard Summary": "Utilizes Claude 3.7 Sonnet for balanced and detailed content summaries, leveraging its strong instruction following and structured output capabilities. Ideal for general articles and web pages.",
        "Comprehensive Analysis": "Powered by Gemini 2.5 Pro for in-depth analysis and insights from complex and long documents, utilizing its superior contextual understanding and ability to handle extensive information.",
        "Efficient Overview": "Relies on o4-mini to quickly provide a concise overview of key points, focusing on essential information for rapid understanding. Best for short content when time is limited.",
        "Technical Breakdown": "Assigns o3 for detailed analysis of technical content, extracting specialized information and implementation details by leveraging its strengths in reasoning and technical explanation.",
        "Research Summary": "Employs Gemini 2.5 Pro for professionally summarizing academic or scientific content, extracting methodologies, findings, and contributions, benefiting from its comprehensive analytical power."
    }
    
    # Map summary_type to actual model based on summarization strengths (user's latest plan)
    model_mapping = {
        "Standard Summary": "claude-3-7-sonnet",  # Strong instruction following and structured output
        "Comprehensive Analysis": "gemini-2-5-pro", # Comprehensive strength for complex tasks
        "Efficient Overview": "o4-mini",            # Best for speed and lightweight tasks
        "Technical Breakdown": "o3",               # Preferred for technical explanation and reasoning
        "Research Summary": "gemini-2-5-pro"      # Comprehensive strength for research content
    }
    
    # Get the actual model from the mapping
    model = model_mapping.get(summary_type, "claude-3-sonnet")
    
    # Create configuration object
    config = KagiConfig(
        model=model,
        internet_access=True  # Internet access must be enabled for webpage summarization
    )
    
    # If instance doesn't exist or model settings changed, recreate instance
    if (_KAGI_INSTANCE is None or _KAGI_INSTANCE.config.model != model):
        _KAGI_INSTANCE = KagiAPI(config)
    
    kagi = _KAGI_INSTANCE
    
    # Webpage summarization requires a new conversation
    kagi.thread_id = None
    
    # Build prompt
    prompt = f"""Please analyze and summarize the content of this webpage: {url}
    
Please include:
1. Overview of main content
2. Key information and points
3. Main arguments or conclusions
4. If technical content, extract key technical details and usage methods
5. If research content, extract main findings and methodology
    
Please present the summary in a clear, structured format."""
    
    result = kagi.send_request(prompt)
    if result:
        return result
    return "Request failed. Please check your network connection, URL validity, or KAGI_COOKIE environment variable."


@mcp.tool()
def kagi_translate(
    text: str = Field(
        description="Text content to be translated",
        examples=[
            "This is a sample text that needs to be translated.",
            "Python is a programming language that lets you work quickly and integrate systems more effectively."
        ]
    ),
    target_language: str = Field(
        description="Target language, e.g.: Chinese, English, Japanese, French, etc.",
        examples=["Chinese", "English", "Japanese", "French", "German", "Spanish"]
    ),
    translation_quality: str = Field(
        description="Select the quality level needed for translation to determine the best model:",
        examples=["Standard Translation", "Technical Translation", "Creative Translation"],
        default="Standard Translation",
        enum=[
            "Standard Translation", "High Accuracy", "Technical Translation", 
            "Quick Translation", "Creative Translation"
        ]
    )
) -> str:
    """Text Translation Tool
    
    Main uses:
    - Translate text from one language to another
    - Maintain the original meaning and tone
    - Suitable for various types of text, including technical documentation, literary works, etc.
    - Support translation between multiple languages
    """
    global _KAGI_INSTANCE
    
    # Translation descriptions for reference
    translation_descriptions = {
        "Standard Translation": "Leverages Claude 3.7 Sonnet for high-quality translation of general text, preserving meaning with good conversational flow and structural integrity. Ideal for everyday content.",
        "High Accuracy": "Powered by Gemini 2.5 Pro for precise translation with nuanced understanding of context and subtle meanings, especially for complex or lengthy texts where utmost accuracy is critical.",
        "Technical Translation": "Assigns o3 for accurate translation of professional or technical content, preserving specialized terminology and technical concepts through its strong analytical and explanatory capabilities.",
        "Quick Translation": "Relies on o4-mini for fast translation of simple content with good efficiency. Focuses on core meaning when speed is essential.",
        "Creative Translation": "Employs Gemini 2.5 Pro to preserve style, tone, and creative elements of the original text, adapting cultural references and maintaining the author's voice for literary or marketing content."
    }
    
    # Map translation_quality to actual model based on translation strengths (user's latest plan)
    model_mapping = {
        "Standard Translation": "claude-3-7-sonnet",# Strong instruction following and structure
        "High Accuracy": "gemini-2-5-pro",         # Comprehensive strength for highest accuracy
        "Technical Translation": "o3",              # Preferred for technical explanations
        "Quick Translation": "o4-mini",               # Best for speed
        "Creative Translation": "gemini-2-5-pro"      # Best for creative content
    }
    
    # Get the actual model from the mapping
    model = model_mapping.get(translation_quality, "claude-3-sonnet")
    
    # Create configuration object
    config = KagiConfig(
        model=model,
        internet_access=True
    )
    
    # If instance doesn't exist or model settings changed, recreate instance
    if (_KAGI_INSTANCE is None or _KAGI_INSTANCE.config.model != model):
        _KAGI_INSTANCE = KagiAPI(config)
    
    kagi = _KAGI_INSTANCE
    
    # Translation requires a new conversation
    kagi.thread_id = None
    
    # Build prompt
    prompt = f"""Please translate the following text to {target_language}, maintaining the original meaning, tone, and format:

{text}

Please only return the translation result, without explanation or additional content."""
    
    result = kagi.send_request(prompt)
    if result:
        return result
    return "Request failed. Please check your network connection or KAGI_COOKIE environment variable."


if __name__ == "__main__":
    # Start MCP service with automatic kagi_chat triggering
    print("Starting Kagi MCP service with automatic chat triggering...")
    print("Use /summarize or /translate commands for specific tools, all other inputs will use kagi_chat")
    mcp.run()