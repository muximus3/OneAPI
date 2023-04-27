from setuptools import setup, find_packages

setup(
    name="one-api-tool",
    version="0.2.5",
    packages=find_packages(),
    package_data={
      "oneapi":["prompt_template/eval_prompt_template.json"]  
    },
    install_requires=[
        # Add your library's dependencies here
        "pydantic",
        "openai",
        "anthropic",
        "requests",
        "httpx",
        "aiohttp",
        "tokenizers",
        "pandas",
        "openpyxl",
        "tabulate",
        "XlsxWriter"
    ],
    entry_points={
        "console_scripts": [
            "one-eval-file=oneapi.commands.one_api_eval_file:main",
            "one-eval-line=oneapi.commands.one_api_eval_one:main",
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