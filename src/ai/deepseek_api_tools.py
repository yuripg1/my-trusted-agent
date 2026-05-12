from typing import Any, NotRequired, Required, TypedDict


class DeepSeekToolFunction(TypedDict):
    name: Required[str]
    description: Required[str]
    parameters: NotRequired[Any]


class DeepSeekTool(TypedDict):
    type: Required[str]
    function: Required[DeepSeekToolFunction]


DEEPSEEK_TOOLS: dict[str, DeepSeekTool] = {
    "execute_bash_command": {
        "type": "function",
        "function": {
            "name": "execute_bash_command",
            "description": 'Run any bash command and return the resulting "stdout", "stderr", and "returncode"',
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The bash command to run"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    "generate_random_integer": {
        "type": "function",
        "function": {
            "name": "generate_random_integer",
            "description": "Generate and return a random integer number",
            "parameters": {
                "type": "object",
                "properties": {
                    "min": {"type": "integer", "description": "The minimum integer (inclusive)"},
                    "max": {"type": "integer", "description": "The maximum integer (inclusive)"},
                },
                "required": ["min", "max"],
                "additionalProperties": False,
            },
        },
    },
    "list_directory": {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List and return the entries in a directory",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "The path of the directory"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return the text contents of a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "The path of the file"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    "read_pdf_document": {
        "type": "function",
        "function": {
            "name": "read_pdf_document",
            "description": "Fetch and return the text contents of a PDF document (local or on the web)",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_type": {
                        "type": "string",
                        "description": 'Type of the location of the PDF document ("local" if it is local or "web" if it is on the web)',
                        "enum": ["local", "web"],
                    },
                    "location": {
                        "type": "string",
                        "description": "Location of the PDF document (local path or web URL)",
                    },
                },
                "required": ["location_type", "location"],
                "additionalProperties": False,
            },
        },
    },
    "read_web_page": {
        "type": "function",
        "function": {
            "name": "read_web_page",
            "description": "Fetch and return the text contents of a web page",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The URL of the web page"}},
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    "search_web": {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": 'Search the web and return results with "title", "href", and "snippet"',
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results_per_page": {
                        "type": "integer",
                        "description": "Maximum number of search results per page",
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "results_page_number": {
                        "type": "integer",
                        "description": "Page number of the search results",
                        "minimum": 1,
                    },
                },
                "required": ["query", "max_results_per_page", "results_page_number"],
                "additionalProperties": False,
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file, creating or overwriting it. Returns the write operation result",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The path of the file to write to"},
                    "content": {"type": "string", "description": "The text content to write to the file"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
}
