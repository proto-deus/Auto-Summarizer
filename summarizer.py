#!/usr/bin/env python3
"""
Document Summarizer Script
Summarizes documents in a folder using an OpenAI-compatible API.
Supports markdown, pdf, and txt files.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Import required libraries
try:
    import requests
except ImportError:
    print("Error: 'requests' library not installed. Run: pip install requests")
    sys.exit(1)

try:
    import PyPDF2
except ImportError:
    print("Warning: 'PyPDF2' not installed. PDF support disabled. Run: pip install PyPDF2")
    PyPDF2 = None


def load_config(config_path: str) -> dict:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def read_text_file(file_path: Path) -> str:
    """Read content from a text or markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def read_pdf_file(file_path: Path) -> str:
    """Read content from a PDF file."""
    if PyPDF2 is None:
        raise ImportError("PyPDF2 is required for PDF support")

    text = []
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text())
    return '\n'.join(text)


def read_document(file_path: Path) -> str:
    """Read document content based on file extension."""
    suffix = file_path.suffix.lower()

    if suffix == '.pdf':
        return read_pdf_file(file_path)
    elif suffix in ['.txt', '.md', '.markdown', '.text']:
        return read_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def summarize_content(content: str, api_config: dict, timeout: int = 300, summary_length: int = None) -> str:
    """Send content to OpenAI-compatible API and get summary."""
    url = api_config.get('url', 'http://localhost:11434/v1/chat/completions')
    api_key = api_config.get('api_key', '')
    model = api_config.get('model', 'llama3.2')

    headers = {
        'Content-Type': 'application/json',
    }
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    # Build the prompt with optional word count target
    word_count_instruction = ""
    if summary_length:
        word_count_instruction = f" The summary should be approximately {summary_length} words long."

    prompt = f"""Please summarize the following document.
Provide a concise summary that captures the main points.{word_count_instruction}

Document content:
{content}

Summary:"""

    data = {
        'model': model,
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'max_tokens': api_config.get('max_tokens', 1000),
        'temperature': api_config.get('temperature', 0.3)
    }

    response = requests.post(url, headers=headers, json=data, timeout=timeout)
    response.raise_for_status()

    result = response.json()
    return result['choices'][0]['message']['content']


def get_supported_files(folder_path: Path, recursive: bool = True) -> list:
    """Get list of supported document files in folder."""
    extensions = ['.txt', '.md', '.markdown', '.text']
    if PyPDF2:
        extensions.append('.pdf')

    files = set()

    if recursive:
        # Recursively search subdirectories
        for ext in extensions:
            files.update(folder_path.rglob(f'*{ext}'))
            files.update(folder_path.rglob(f'*{ext.upper()}'))
    else:
        # Only search in the specified folder (no subdirectories)
        for ext in extensions:
            files.update(folder_path.glob(f'*{ext}'))
            files.update(folder_path.glob(f'*{ext.upper()}'))

    return sorted(files)


def get_relative_path(file_path: Path, base_path: Path) -> str:
    """Get relative path from base folder for display purposes."""
    try:
        rel_path = file_path.relative_to(base_path)
        # If file is in a subdirectory, include the path
        if rel_path.parent != Path('.'):
            return str(rel_path.parent / rel_path.stem)
        return rel_path.stem
    except ValueError:
        return file_path.stem


def main():
    parser = argparse.ArgumentParser(
        description='Summarize documents in a folder using an OpenAI-compatible API'
    )

    # Configuration file options
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to config.json file'
    )

    # Input/Output options
    parser.add_argument(
        '--folder', '-f',
        type=str,
        help='Folder containing documents to summarize (default: current directory)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output markdown file path (default: summaries.md)'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Disable recursive search in subfolders (default: recursive enabled)'
    )

    # API options
    parser.add_argument(
        '--url',
        type=str,
        help='API endpoint URL (e.g., http://localhost:11434/v1/chat/completions)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key for authentication'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Model name to use (e.g., llama2, gpt-3.5-turbo)'
    )

    # API behavior options
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        help='API request timeout in seconds (default: 300)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        dest='max_tokens',
        help='Maximum tokens in API response (default: 1000)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        help='API temperature setting (default: 0.3)'
    )
    parser.add_argument(
        '--max-content-length',
        type=int,
        dest='max_content_length',
        help='Maximum characters to send to API, content is truncated if longer (default: 10000)'
    )

    # Summary options
    parser.add_argument(
        '--summary-length',
        type=int,
        dest='summary_length',
        help='Target summary length in words'
    )

    args = parser.parse_args()

    # Load configuration from file if specified
    api_config = {}
    if args.config:
        if os.path.exists(args.config):
            api_config = load_config(args.config)
        else:
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)

    # Override config with command line arguments (CLI takes precedence)
    if args.folder:
        api_config['folder'] = args.folder
    if args.url:
        api_config['url'] = args.url
    if args.api_key:
        api_config['api_key'] = args.api_key
    if args.model:
        api_config['model'] = args.model
    if args.timeout:
        api_config['timeout'] = args.timeout
    if args.summary_length:
        api_config['summary_length'] = args.summary_length
    if args.output:
        api_config['output'] = args.output
    if args.max_tokens:
        api_config['max_tokens'] = args.max_tokens
    if args.temperature is not None:
        api_config['temperature'] = args.temperature
    if args.max_content_length:
        api_config['max_content_length'] = args.max_content_length

    # Set defaults for any missing values
    api_config.setdefault('url', 'http://localhost:11434/v1/chat/completions')
    api_config.setdefault('timeout', 300)
    api_config.setdefault('output', 'summaries.md')
    api_config.setdefault('max_tokens', 1000)
    api_config.setdefault('temperature', 0.3)
    api_config.setdefault('max_content_length', 10000)

    # Determine if we should search recursively
    recursive = not args.no_recursive
    if 'recursive' in api_config:
        recursive = api_config['recursive']

    # Validate required configuration
    if 'url' not in api_config:
        print("Error: API URL must be provided via --url or config.json")
        sys.exit(1)

    # Get folder path (default to current directory if not specified)
    folder_arg = api_config.get('folder', '.')
    folder_path = Path(folder_arg).resolve()
    if not folder_path.is_dir():
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    # Get supported files
    files = get_supported_files(folder_path, recursive=recursive)
    if not files:
        print(f"No supported files found in {folder_path}")
        if recursive:
            print("  (try --no-recursive if you want to search only the root folder)")
        sys.exit(0)

    summary_length = api_config.get('summary_length')

    print(f"Found {len(files)} file(s) to summarize")
    print(f"Search mode: {'recursive' if recursive else 'non-recursive'}")
    print(f"Timeout: {api_config['timeout']} seconds")
    print(f"Max tokens: {api_config['max_tokens']}")
    print(f"Temperature: {api_config['temperature']}")
    if summary_length:
        print(f"Target summary length: ~{summary_length} words")

    # Process each file
    results = []
    for file_path in files:
        print(f"\nProcessing: {file_path}")

        try:
            content = read_document(file_path)

            # Truncate if content is too long
            max_length = api_config.get('max_content_length', 10000)
            if len(content) > max_length:
                content = content[:max_length] + "...[truncated]"

            summary = summarize_content(
                content,
                api_config,
                timeout=api_config['timeout'],
                summary_length=summary_length
            )

            # Get relative path for the title
            title = get_relative_path(file_path, folder_path)

            results.append({
                'title': title,
                'summary': summary,
                'file': str(file_path)
            })
            print(f"  ✓ Summarized successfully")

        except requests.exceptions.Timeout:
            print(f"  ✗ Timeout error: API took longer than {api_config['timeout']} seconds")
            results.append({
                'title': get_relative_path(file_path, folder_path),
                'summary': f"Error: Timeout after {api_config['timeout']} seconds",
                'file': str(file_path)
            })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'title': get_relative_path(file_path, folder_path),
                'summary': f"Error: {str(e)}",
                'file': str(file_path)
            })

    # Output results
    output_lines = []
    for result in results:
        output_lines.append(f"## {result['title']}\n")
        output_lines.append(f"{result['summary']}\n")
        output_lines.append("---\n")

    output_text = '\n'.join(output_lines)

    # Write to output file
    output_path = Path(api_config['output'])
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_text)
    print(f"\nResults written to: {output_path.absolute()}")


if __name__ == '__main__':
    main()

