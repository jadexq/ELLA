---
title: Using a Coding Agent
layout: default
nav_order: 3
---

## Using a Coding Agent

ELLA ships an [`AGENTS.md`](https://github.com/jadexq/ELLA/blob/main/AGENTS.md), an instruction file for coding agents (Claude Code, Cursor, Codex, etc.). It gives an agent everything needed to run ELLA on your data: the install command, the full pipeline recipe, the input and output contract, common pitfalls, and a smoke test.

Agents that support the `AGENTS.md` convention read it automatically when working in this repo. Otherwise, just point your agent at `AGENTS.md` (for example: "read AGENTS.md and run ELLA on my data").
