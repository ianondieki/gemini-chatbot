"""Angel - a deliberate Gemini agent, in your terminal.

Every turn runs the same loop: plan the work, take steps with tools, review the
draft against the plan, and revise if the review fails. The trace of that is
printed as it happens, so you can watch the agent think rather than guess at it.

    python chatbot.py

Type /help once you are in for the commands.

The agent itself lives in the ``gemini_agent`` package - this file is only a
front end. ``app.py`` is the same agent behind a browser UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from gemini_agent import AgentSession, ConfigError, ConsoleRenderer
from gemini_agent.render import Style

load_dotenv()

BANNER = r"""
    _                    _
   / \   _ __   __ _  __| |
  / _ \ | '_ \ / _` |/ _` |
 / ___ \| | | | (_| | (_| |
/_/   \_\_| |_|\__, |\__,_|
               |___/
"""

COMMANDS = """
  /help              this list
  /clear             forget the conversation (documents stay loaded)
  /history           print the conversation so far
  /tools             list the tools the agent currently has
  /notes             show what the agent has chosen to remember
  /stats             model, token usage and session state
  /load <path.pdf>   index a PDF so the agent can search it
  /unload            drop the loaded documents
  /plan <mode>       planning: auto | always | never
  /trace             toggle the live reasoning trace
  /thoughts          toggle the model's thinking summaries
  /quit              leave
"""


def _fatal(style: Style, message: str) -> int:
    print(style.red(f"\n{message}\n"))
    return 1


def _welcome(session: AgentSession, style: Style) -> None:
    print(style.magenta(BANNER))
    print(f"  {style.bold('Model')}    {session.config.model}")
    print(f"  {style.bold('Tools')}    {', '.join(session.tool_names)}")
    print(
        f"  {style.bold('Loop')}     plan={session.config.planning}, "
        f"up to {session.config.max_steps} steps and "
        f"{session.config.max_reflections} review pass(es) per turn"
    )
    print(style.dim("\n  /help for commands, /quit to leave\n"))


def _handle_command(session: AgentSession, renderer: ConsoleRenderer, line: str) -> bool:
    """Run a slash command. Returns False when it is time to quit."""
    style = renderer.style
    parts = line[1:].split(maxsplit=1)
    command = parts[0].lower() if parts else ""
    argument = parts[1].strip() if len(parts) > 1 else ""

    if command in ("quit", "exit", "q"):
        return False

    if command == "help":
        print(COMMANDS)

    elif command == "clear":
        session.clear()
        print(style.dim("  conversation cleared\n"))

    elif command == "history":
        transcript = session.transcript()
        print(f"\n{transcript}\n" if transcript else style.dim("  nothing yet\n"))

    elif command == "tools":
        print()
        for name in session.tool_names:
            spec = session.registry.get(name)
            print(f"  {style.cyan(name)}({spec.signature_hint})")
            print(style.dim(f"      {spec.description[:160]}"))
        print()

    elif command == "notes":
        notes = session.memory.notes
        if not notes:
            print(style.dim("  no notes saved yet\n"))
        else:
            print()
            for note in notes:
                print(f"  - {note.render()}")
            print()

    elif command == "stats":
        print()
        for key, value in session.stats().items():
            print(f"  {key:<16} {value}")
        print()

    elif command == "load":
        _load_pdf(session, style, argument)

    elif command == "unload":
        session.unload_documents()
        print(style.dim("  documents dropped\n"))

    elif command == "plan":
        if argument in ("auto", "always", "never"):
            session.reconfigure(planning=argument)
            print(style.dim(f"  planning set to {argument}\n"))
        else:
            print(style.yellow("  usage: /plan auto | always | never\n"))

    elif command == "trace":
        renderer.verbose = not renderer.verbose
        print(style.dim(f"  trace {'on' if renderer.verbose else 'off'}\n"))

    elif command == "thoughts":
        renderer.show_thoughts = not renderer.show_thoughts
        print(style.dim(f"  thoughts {'on' if renderer.show_thoughts else 'off'}\n"))

    else:
        print(style.yellow(f"  unknown command '{command}' - try /help\n"))

    return True


def _load_pdf(session: AgentSession, style: Style, argument: str) -> None:
    if not argument:
        print(style.yellow("  usage: /load path/to/file.pdf\n"))
        return

    path = Path(argument).expanduser()
    if not path.is_file():
        print(style.red(f"  no such file: {path}\n"))
        return

    print(style.dim(f"  indexing {path.name}..."))
    last = [-1]

    def progress(done: int, total: int) -> None:
        percent = int(done * 100 / max(1, total))
        if percent // 10 != last[0] // 10:
            last[0] = percent
            print(style.dim(f"    {done}/{total} chunks embedded"), flush=True)

    try:
        with path.open("rb") as handle:
            added = session.load_pdf(handle, path.name, progress)
    except Exception as exc:
        print(style.red(f"  could not index it: {exc}\n"))
        return

    print(style.green(f"  indexed {added} chunks - search_documents is now available\n"))


def main() -> int:
    style = Style()
    try:
        session = AgentSession.create(name="Angel")
    except ConfigError as exc:
        return _fatal(style, f"{exc}")
    except Exception as exc:
        return _fatal(style, f"Could not start: {type(exc).__name__}: {exc}")

    renderer = ConsoleRenderer()
    _welcome(session, style)

    if session.search_client is None:
        print(
            style.yellow(
                "  note: TAVILY_API_KEY is not set, so the agent has no web "
                "access this session.\n"
            )
        )

    while True:
        try:
            line = input(style.bold("you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            return 0

        if not line:
            continue

        if line.startswith("/"):
            if not _handle_command(session, renderer, line):
                print()
                return 0
            continue

        # Bare quit/exit still works, out of habit.
        if line.lower() in ("quit", "exit"):
            print()
            return 0

        print()
        try:
            result = session.ask(line, sink=renderer)
        except KeyboardInterrupt:
            print(style.yellow("\n  interrupted\n"))
            continue

        print(f"\n{style.bold('angel>')} {result.answer}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
