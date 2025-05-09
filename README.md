# Kagi MCP - Intelligent Search Assistant

Kagi MCP is an intelligent assistant based on the Model Control Protocol (MCP) that integrates Kagi's AI services to provide high-quality search-driven conversations, webpage summaries, and text translation capabilities. This tool leverages multiple advanced AI models, automatically selecting the most appropriate model for different task types to ensure optimal response quality.

## Features

### 1. Intelligent Conversation (kagi_chat)

Automatically triggered search-driven intelligent assistant that can:
- Answer almost any question with the latest, most accurate information
- Search for technical documentation and tutorials
- Troubleshoot common errors and issues
- Find configuration guides for software and tools
- Search for code examples of specific functionality
- Get recommended development best practices
- Stay updated on the latest developments in technology and industries

### 2. Webpage Summarization (kagi_summarize)

Quickly analyze and summarize webpage content:
- Extract main content from long articles
- Extract key information from webpages
- Get main points and conclusions from articles
- Save reading time
- Analyze technical documentation and open source projects
- Extract key findings from papers and research reports

### 3. Text Translation (kagi_translate)

High-quality text translation tool:
- Translate text from one language to another
- Maintain the original meaning and tone
- Suitable for various types of text, including technical documentation, literary works, etc.
- Support translation between multiple languages

## Intelligent Model Selection

The system automatically selects the most appropriate AI model based on the task type:

### Conversation Model Selection
- **General Knowledge**: Optimized for general questions (claude-3-7-sonnet)
- **Advanced Reasoning**: For complex reasoning problems (o3)
- **Balanced Performance**: Provides balance between speed and quality (claude-3-7-sonnet)
- **Creative Content**: For creative content generation (gemini-2-5-pro)
- **Technical Analysis**: For technical analysis questions (o3)
- **Architecture Design**: For system architecture discussions (o3)
- **Quick Response**: For quick factual questions (o4-mini)
- **Code Generation**: For programming and code-related questions (gemini-2-5-pro)
- **Scientific Research**: For scientific research questions (gemini-2-5-pro)

### Summary Model Selection
- **Standard Summary**: Provides balanced and detailed content summary (claude-3-7-sonnet)
- **Comprehensive Analysis**: Delivers in-depth analysis and insights (gemini-2-5-pro)
- **Efficient Overview**: Quickly provides a concise overview of key points (o4-mini)
- **Technical Breakdown**: Analyzes technical content in detail (o3)
- **Research Summary**: Professionally summarizes academic or scientific content (gemini-2-5-pro)

### Translation Model Selection
- **Standard Translation**: Provides high-quality translation for general text (claude-3-7-sonnet)
- **High Accuracy**: Delivers precise translation for formal documents (gemini-2-5-pro)
- **Technical Translation**: Accurate translation of professional or technical content (o3)
- **Quick Translation**: Fast translation for simple content (o4-mini)
- **Creative Translation**: Preserves style, tone, and creative elements of the original text (gemini-2-5-pro)

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Kagi account and valid Cookie

### Installation Steps

1. Clone the repository or download the source code

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables
   Create a `.env` file in the project root directory and add your Kagi Cookie:
   ```
   KAGI_COOKIE=your_kagi_cookie_here
   ```
   
   How to get your Kagi Cookie:
   - Log in to the Kagi website (https://kagi.com)
   - Open browser developer tools (F12)
   - Go to the Network tab
   - Refresh the page
   - Find any request and look for the Cookie value in Headers
   - Copy the entire Cookie string

## Usage

### Starting the Service
```bash
python kagi.py
```

### Using Commands
- Regular questions will automatically use the `kagi_chat` tool
- Use the `/summarize [URL]` command to summarize webpage content
- Use the `/translate [text]` command to translate text

### Examples

1. Regular conversation (automatically uses kagi_chat):
   ```
   How to process JSON data in Python?
   ```

2. Webpage summarization:
   ```
   /summarize https://en.wikipedia.org/wiki/Artificial_intelligence
   ```

3. Text translation:
   ```
   /translate Python is a programming language that lets you work quickly and integrate systems more effectively. Chinese
   ```

## Advanced Features

### Caching Mechanism
The system implements a caching mechanism that can cache responses in non-session mode, improving response speed and reducing API calls.

### Session Management
The system automatically manages session context to maintain conversation coherence. You can start a new conversation by setting `new_conversation=True`.

### Custom Configuration
You can customize API configuration by modifying the `KagiConfig` class, such as timeout, user agent, etc.

## Technical Architecture

This project is built on the following technologies:
- **MCP (Model Control Protocol)**: For building and managing AI tools
- **FastMCP**: Fast implementation of MCP for creating AI services
- **Kagi API**: Provides high-quality AI responses and search capabilities
- **Requests**: For HTTP requests
- **HTML2Text**: For converting HTML to Markdown format
- **Python-dotenv**: For environment variable management

## Important Notes

- A valid Kagi Cookie is required to use the service
- Cookies have a limited validity period and need to be updated after expiration
- Please comply with Kagi's terms of use and limitations when using the API

## Contributions and Feedback

Contributions, issue reports, and feature requests are welcome. Please submit your feedback and contributions through GitHub Issues or Pull Requests.

## License

[MIT License](LICENSE)