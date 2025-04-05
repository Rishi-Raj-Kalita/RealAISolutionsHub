# Web Agent Automation

This project demonstrates an AI agent that can autonomously navigate web interfaces to complete tasks. The example implementation shows an agent ordering a cheesecake from Swiggy without human intervention.

## Features

- Autonomous browser control through an AI agent
- Support for multiple LLM providers (local Ollama models, AWS Bedrock)
- Visual understanding capabilities using browser screenshots
- Conversation logging for debugging and analysis
- Observation tracking with Laminar.ai


## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Rishi-Raj-Kalita/.git
   cd browser-use-swiggy
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with the following variables:
   ```
   ACCESS_KEY=your_aws_access_key
   SECRET_KEY=your_aws_secret_key
   LAMINAR=your_laminar_api_key
   ```

## Usage

Run the agent with the default configuration:
```
python main.py
```

### Configuration Options

The agent can be configured to use different LLM models:

- Local models via Ollama:
  ```python
  llm = get_model(model='deepseek-r1:7b', provider='local')
  ```

- Llama model via Ollama:
  ```python
  llm = get_model(provider='llama')
  ```

- AWS Bedrock models:
  ```python
  llm = get_model(model='anthropic.claude-3-5-sonnet-20240620-v1:0', provider='aws')
  ```

### Customizing Tasks

Define tasks as a series of steps in natural language:

```python
tasks = """
1. Click on Search box of landing page.
2. Type "Restaurant Name" in the search box and click on the search icon.
...
"""
```

## How It Works

1. The agent initializes a Chrome browser session
2. It reads the task description to understand what needs to be done
3. For each step, it:
   - Captures a screenshot of the current page
   - Analyzes the visual content to identify UI elements
   - Decides on the next action based on the task description
   - Performs the action (clicking, typing, etc.)
4. The entire process is monitored through Laminar.ai for debugging and analysis

## Dependencies

- browser_use: For browser automation capabilities
- langchain_ollama: Integration with local Ollama models
- langchain_aws: Integration with AWS Bedrock models
- lmnr: Laminar.ai client library for observability
- dotenv: Environment variable management
- boto3: AWS SDK for Python

## License

[Your license information here]

## Contributing

[Your contribution guidelines here]