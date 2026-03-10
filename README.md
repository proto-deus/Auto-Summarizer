# Auto Summarizer

A Python script that summarizes documents in a folder (including subfolders) using an OpenAI-compatible API. Supports markdown, PDF, and text files.

## Features

- **Multiple file formats**: Process `.md`, `.pdf` and `.txt` files
- **Recursive search**: Can automatically find all documents in subfolders
- **Flexible configuration**: Use CLI arguments, config.json, or both
- **Customizable summaries**: Set target word count, max tokens, and temperature
- **Markdown output**: Results formatted as titles with summaries in Markdown
- **Tagging**: Add tags to the summaries either maunally or automaticaly by the model.

## Installation

Install the requirements

```bash
pip install requests PyPDF2
```

## Quick Start

```bash
# Using a config file
python summarizer.py --config config.json

# Using CLI arguments only
python summarizer.py --url http://localhost:11434/v1/chat/completions --model llama3.2 --folder ./docs
```

## Configuration

### Config File (config.json)

```json
{
    "url": "http://localhost:11434/v1/chat/completions",
    "api_key": "your-api-key-here",
    "model": "llama3.2",
    "timeout": 300,
    "max_tokens": 1000,
    "temperature": 0.3,
    "max_content_length": 10000,
    "summary_length": 150,
    "folder": "./docs",
    "output": "summaries.md",
    "recursive": true
    "tags": [],
    "auto_tag": true,
    "num_tags": 10
}
```

### CLI Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-c, --config` | Path to config.json | (none) |
| `-f, --folder` | Folder with documents | `.` |
| `-o, --output` | Output markdown file | `summaries.md` |
| `--no-recursive` | Disable subfolder search | recursive enabled |
| `--url` | API endpoint URL | localhost:11434 |
| `--api-key` | API authentication key | (none) |
| `-m, --model` | Model name | llama3.2 |
| `-t, --timeout` | Request timeout (seconds) | 300 |
| `--max-tokens` | Max response tokens | 1000 |
| `--temperature` | API temperature (0-2) | 0.3 |
| `--max-content-length` | Max chars sent to API | 10000 |
| `--summary-length` | Target summary words | (none) |
| `--tag` | Adds a tag to the summary | (none) |
| `--auto_tag` | Has the LLM add tags to the summary | true |
| `--num_tags` | Number of tags for the LLM to generate | 5 |

## Output Format

```markdown
## document-name

Summary of the document content...

---

## subfolder/another-document

Another summary...

[TAGS: tag1, tag2, tag3]

---
```

## Requirements

- Python 3.7+
- `requests` library
- `PyPDF2` library (for PDF support)
