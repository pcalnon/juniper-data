# MCP Servers, Setup, Config, and Startup

## URL For SERENA docs

<https://oraios.github.io/serena/02-usage/030_clients.html>

## Installation

### Install onto host via Mamba

```bash
mamba install uv

uvx --from git+https://github.com/oraios/serena serena start-mcp-server --help

```

### Install into claude code

```bash
claude mcp add serena -- uvx --from git+https://github.com/oraios/serena serena start-mcp-server --context claude-code --project "$(pwd)"
```

---
