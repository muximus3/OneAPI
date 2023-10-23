from setuptools import setup, find_packages

setup(
    name="one-api-tool",
    version="1.1.9",
    packages=find_packages(),
    install_requires=[
        # Add your library's dependencies here
        "pydantic",
        "openai",
        "anthropic>=0.3",
        "huggingface_hub",
        "requests",
        "httpx",
        "rich",
        "inquirer",
        "Jinja2",
        "aiohttp",
        "tiktoken",
        "python-dotenv",
        "tqdm",
        "tokenizers",
        "docstring_parser"
    ],
    entry_points={
        "console_scripts": [
            "one-api=oneapi.commands.one_api_requst:main"
        ]
    },
    description="Use only one line of code to call multiple model APIs similar to ChatGPT. Currently supported: Azure OpenAI Resource endpoint API, OpenAI Official API, and Anthropic Claude series model API.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/muximus3/OneAPI",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)