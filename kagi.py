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
        "General Knowledge": "Optimized for general questions and information retrieval. Provides comprehensive and accurate answers with balanced context. Best for everyday queries, factual information, and broad topics. (claude-3-sonnet)",
        "Advanced Reasoning": "Specialized in complex reasoning and multi-step analysis. Delivers detailed reasoning processes and handles nuanced problems. Ideal for decision-making, logical analysis, and complex problem-solving. (claude-3-7-sonnet)",
        "Balanced Performance": "Offers an excellent balance between speed and quality. Provides consistent results with good efficiency. Perfect for daily use when you need reliable answers without maximum depth. (claude-3-5-sonnet)",
        "Creative Content": "Excels at creative writing and diverse content generation. Produces unique expressions and varied outputs. Best for storytelling, marketing copy, and content that requires originality. (gemini-2-5-pro)",
        "Technical Analysis": "Focused on precise technical understanding and explanation. Delivers accurate technical insights and complex concept clarification. Ideal for technical documentation, error analysis, and specialized knowledge. (gpt-4-1)",
        "Architecture Design": "Specialized in software architecture and system design. Excels at analyzing entire codebases and providing structural insights. Perfect for system planning, architecture reviews, and technical design. (gemini-2-5-pro)",
        "Quick Response": "Optimized for fast, efficient responses to simple questions. Delivers concise answers with minimal latency. Best for quick facts, simple definitions, and time-sensitive queries. (o4-mini)",
        "Code Generation": "Focused on programming and software development. Provides consistent, well-structured code with minimal prompting. Ideal for coding assistance, debugging, and software implementation. (claude-3-5-sonnet)",
        "Scientific Research": "Specialized in academic research and scientific analysis. Delivers in-depth exploration of specialized domains. Perfect for research papers, scientific literature review, and academic writing. (deepseek)"
    }
    
    # Map model_selection to actual model based on their strengths (based on latest research)
    model_mapping = {
        "General Knowledge": "claude-3-sonnet",      # Excellent general capabilities and balanced performance
        "Advanced Reasoning": "claude-3-7-sonnet",   # Superior reasoning abilities, ideal for complex analysis
        "Balanced Performance": "claude-3-5-sonnet", # Good balance of speed and quality
        "Creative Content": "gemini-2-5-pro",        # Creative generation and diverse outputs
        "Technical Analysis": "gpt-4-1",             # Precise technical understanding and explanation
        "Architecture Design": "gemini-2-5-pro",     # Excels at analyzing entire codebases and system design
        "Quick Response": "o4-mini",                 # More efficient responses than o3-mini
        "Code Generation": "claude-3-5-sonnet",      # Excellent at programming tasks with consistent results
        "Scientific Research": "deepseek"            # Specialized academic and research capabilities
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
        "Standard Summary": "Provides a balanced and detailed content summary. Extracts key information while maintaining context. Ideal for general articles, blog posts, and web pages. (claude-3-sonnet)",
        "Comprehensive Analysis": "Delivers in-depth analysis and insights. Identifies patterns, implications, and connections within complex content. Perfect for long documents, research papers, and detailed reports. (claude-3-7-sonnet)",
        "Efficient Overview": "Quickly provides a concise overview of key points. Focuses on essential information for rapid understanding. Best for news articles, short content, and when time is limited. (o4-mini)",
        "Technical Breakdown": "Analyzes technical content in detail. Extracts specialized information, implementation details, and technical specifications. Ideal for documentation, code repositories, and technical guides. (gpt-4-1)",
        "Research Summary": "Professionally summarizes academic or scientific content. Extracts research methodologies, findings, and contributions. Perfect for academic papers, scientific literature, and specialized research. (deepseek)"
    }
    
    # Map summary_type to actual model based on summarization strengths
    model_mapping = {
        "Standard Summary": "claude-3-sonnet",        # Balanced summarization capabilities for general content
        "Comprehensive Analysis": "claude-3-7-sonnet", # Deep analysis and insight capabilities
        "Efficient Overview": "o4-mini",              # Fast and efficient overview, suitable for simple content
        "Technical Breakdown": "gpt-4-1",             # Precise technical content analysis
        "Research Summary": "deepseek"                # Professional academic and research content summarization
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
        "Standard Translation": "Provides high-quality translation for general text while preserving original meaning. Balances accuracy and natural flow. Ideal for everyday content, correspondence, and general documents. (claude-3-sonnet)",
        "High Accuracy": "Delivers precise translation with nuanced understanding of context and subtle meanings. Maintains formal tone and structure. Perfect for official documents, legal texts, and content where precision is critical. (claude-3-7-sonnet)",
        "Technical Translation": "Specializes in accurate translation of professional or technical content. Preserves specialized terminology and technical concepts. Best for technical documentation, scientific papers, and specialized fields. (gpt-4-1)",
        "Quick Translation": "Offers fast translation for simple content with good efficiency. Focuses on core meaning rather than nuance. Ideal for casual conversations, short texts, and when speed is essential. (o4-mini)",
        "Creative Translation": "Preserves style, tone, and creative elements of the original text. Adapts cultural references and maintains the author's voice. Perfect for literary works, marketing content, and creative writing. (gemini-2-5-pro)"
    }
    
    # Map translation_quality to actual model based on translation strengths
    model_mapping = {
        "Standard Translation": "claude-3-sonnet",    # Good general translation capabilities
        "High Accuracy": "claude-3-7-sonnet",         # Most precise translation, suitable for important documents
        "Technical Translation": "gpt-4-1",           # Accurate translation of specialized terminology and technical content
        "Quick Translation": "o4-mini",               # More efficient quick translation than o3-mini
        "Creative Translation": "gemini-2-5-pro"      # Creative translation that preserves style and tone
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