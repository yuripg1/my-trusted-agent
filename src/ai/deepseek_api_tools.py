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
    "delete_path": {
        "type": "function",
        "function": {
            "name": "delete_path",
            "description": "Delete a file, an empty directory, or a symlink",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": 'The type of the path: "file", "directory", or "symlink"',
                        "enum": ["file", "directory", "symlink"],
                    },
                    "path": {
                        "type": "string",
                        "description": "Path of the file, empty directory, or symlink to delete",
                    },
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
            "description": "Edit a file by searching for an exact piece of text and replacing it with another piece of text",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file to edit"},
                    "search_for": {"type": "string", "description": "The exact text to search for"},
                    "replace_with": {"type": "string", "description": "The text to replace occurrences with"},
                    "number_of_substitutions": {
                        "type": "integer",
                        "description": "Exact number of substitutions expected to be performed (return an error if it does not match the number of occurrences found)",
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
            "description": "Execute any shell command and return the resulting stdout, stderr, and exit code (stdout and stderr are returned without the need for redirection; you should strongly prefer using the other available tools instead of executing shell commands; treat shell command execution as a powerful last resort to be used when the other available tools do not suffice)",
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
    "move_path": {
        "type": "function",
        "function": {
            "name": "move_path",
            "description": "Move or rename a file, directory, or symlink from source to destination",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": 'The type of the path: "file", "directory", or "symlink"',
                        "enum": ["file", "directory", "symlink"],
                    },
                    "source": {"type": "string", "description": "Current path of the file, directory, or symlink"},
                    "destination": {"type": "string", "description": "New path for the file, directory, or symlink"},
                },
                "required": ["type", "source", "destination"],
                "additionalProperties": False,
            },
        },
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return the text contents of a file (in full or partially)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file"},
                    "start_line": {
                        "type": "integer",
                        "description": "Start reading from this line number (inclusive; defaults to the first line)",
                        "minimum": 1,
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Stop reading at this line number (inclusive; defaults to the last line)",
                        "minimum": 1,
                    },
                },
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
                        "default": 1,
                    },
                },
                "required": ["query", "max_results_per_page"],
                "additionalProperties": False,
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text contents to a file (if the file does not exist, create it; if the file exist, either overwrite it or append to it, depending on the chosen mode; if the directory does not exist, create it; if the parent directories do not exist, create them)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file"},
                    "mode": {
                        "type": "string",
                        "description": 'The write mode ("create_or_overwrite" to overwrite if exists or create otherwise; "create_if_not_exists" to create only if the file does not already exist; "append" to append to the end of the file if it exists or create it otherwise)',
                        "enum": ["create_or_overwrite", "create_if_not_exists", "append"],
                    },
                    "content": {"type": "string", "description": "Text contents to write"},
                },
                "required": ["path", "mode", "content"],
                "additionalProperties": False,
            },
        },
    },
}
