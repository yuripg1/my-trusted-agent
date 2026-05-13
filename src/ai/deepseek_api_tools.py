from typing import Any, NotRequired, Required, TypedDict


class DeepSeekToolFunction(TypedDict):
    name: Required[str]
    description: Required[str]
    parameters: NotRequired[Any]


class DeepSeekTool(TypedDict):
    type: Required[str]
    function: Required[DeepSeekToolFunction]


DEEPSEEK_TOOLS: dict[str, DeepSeekTool] = {
    "create_directory": {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Create a directory (if the parent directories do not exist, create them)",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path of the directory to be created"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    "delete_file_or_directory": {
        "type": "function",
        "function": {
            "name": "delete_file_or_directory",
            "description": "Delete a file or an empty directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": 'The type of the path ("file" or "directory")',
                        "enum": ["file", "directory"],
                    },
                    "path": {"type": "string", "description": "Path of the file or directory"},
                },
                "required": ["type", "path"],
                "additionalProperties": False,
            },
        },
    },
    "edit_file": {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by searching for specific piece of text and replacing it with another piece of text (return an error if the number of occurrences found does not exactly match the expected number of substitutions)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file to edit"},
                    "search_for": {"type": "string", "description": "The exact text to search for"},
                    "replace_with": {"type": "string", "description": "The text to replace occurrences with"},
                    "number_of_substitutions": {
                        "type": "integer",
                        "description": "Exact number of substitutions expected to be performed",
                        "minimum": 1,
                    },
                },
                "required": ["path", "search_for", "replace_with", "number_of_substitutions"],
                "additionalProperties": False,
            },
        },
    },
    "execute_shell_command": {
        "type": "function",
        "function": {
            "name": "execute_shell_command",
            "description": "Execute any shell command and return the resulting stdout, stderr, and exit code",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The command to be executed"}},
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
                "properties": {"path": {"type": "string", "description": "Path of the file"}},
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
            "description": "Write text contents to a file (if the file does not exist, create it; if the file exist, overwrite it; if the directory does not exist, create it; if the parent directories do not exist, create them)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file"},
                    "content": {"type": "string", "description": "Text contents to write"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
}
