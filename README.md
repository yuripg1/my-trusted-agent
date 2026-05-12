# My trusted AI agent

This projects aims at implementing a minimalistic terminal-based AI agent that is cheap to run.

---

## Supported AI models

* DeepSeek
    * deepseek-v4-flash
    * deepseek-v4-pro

---

## Special commands

* **/new** : Starts a new session
* **/raw** : Starts a new raw session (without any system prompts)
* **/load n** : Loads previous session with ID "n"
* **/replay** : Replays a chat (useful after loading a session)
* **/rewind** : Rewinds back to right before the latest user prompt
* **/exit** : Closes the application

---

## Security

The AI agent does have support for tool calling.

However, for security reasons, the following actions require manual user permission:

* Execution of shell commands
* Reading of local PDF documents
* File reading
* File writing

---

## How to run

### Setup

```shell
$ sudo apt update
$ sudo apt install -y python3-dev python3-pip python3-venv
```

### Run

```shell
$ bash ./run.sh
```
