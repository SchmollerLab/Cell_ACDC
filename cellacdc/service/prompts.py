"""Client-side prompts.

Anything that needs the user (choose a PyTorch build, confirm an install, ...)
is described as a small, JSON-serializable `Prompt` and answered by a
*handler*. The code asking does not know whether the answer comes from a Qt
dialog, the terminal or a default, and prompts can be created from any thread.

Writing a new prompt::

    from cellacdc.service import prompts

    answer = ask(prompts.choice(
        'Install PyTorch', 'Which build?', ['CPU', 'CUDA 12.8'], default='CPU'
    ))                                  # -> str, or None if cancelled
    if ask(prompts.confirm('Install', 'Install package X?')):   # -> bool
        ...

where `ask` is the `prompt_func` received by the environment functions (or
`QtHandler(parent).ask`, `CliHandler().ask`, `AutoHandler().ask`).

Answers: choice -> selected label or None, confirm -> bool, text -> str or
None, info -> None.

This module must not import Qt at module level (the worker is lightweight).
"""

import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, List, Optional

CHOICE = 'choice'
CONFIRM = 'confirm'
TEXT = 'text'
INFO = 'info'


@dataclass
class Prompt:
    kind: str
    title: str
    text: str = ''
    choices: Optional[List[str]] = None
    default: Any = None
    details: Optional[str] = None
    commands: List[str] = field(default_factory=list)
    ok_text: str = 'Ok'
    cancel_text: str = 'Cancel'
    # Optional hint for a specialised client dialog (e.g. 'pytorch'); 
    # handlers that don't know it fall back to the generic widget
    dialog: Optional[str] = None
    extra: dict = field(default_factory=dict)
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Prompt':
        return cls(**data)


def choice(title, text, choices, default=None, **kwargs) -> Prompt:
    return Prompt(CHOICE, title, text, list(choices), default, **kwargs)


def confirm(title, text, default=True, **kwargs) -> Prompt:
    return Prompt(CONFIRM, title, text, default=default, **kwargs)


def text_input(title, text, default='', **kwargs) -> Prompt:
    return Prompt(TEXT, title, text, default=default, **kwargs)


def info(title, text, **kwargs) -> Prompt:
    return Prompt(INFO, title, text, **kwargs)


def _default_answer(prompt: Prompt):
    if prompt.kind == CHOICE:
        if prompt.default in (prompt.choices or []):
            return prompt.default
        return prompt.choices[0] if prompt.choices else None
    if prompt.kind == CONFIRM:
        return bool(prompt.default)
    if prompt.kind == TEXT:
        return prompt.default
    return None


class PromptHandler:
    """Base class. Subclasses implement `ask(prompt) -> answer`."""

    def ask(self, prompt: Prompt):
        raise NotImplementedError

    __call__ = lambda self, prompt: self.ask(prompt)


class AutoHandler(PromptHandler):
    """Never asks: returns the default answer (for headless/automatic runs)."""

    def ask(self, prompt: Prompt):
        return _default_answer(prompt)


class CliHandler(PromptHandler):
    """Asks in the terminal."""

    def __init__(self, input_func: Callable = input, print_func: Callable = print):
        self._input = input_func
        self._print = print_func

    def ask(self, prompt: Prompt):
        self._print('-' * 60)
        self._print(prompt.title)
        if prompt.text:
            self._print(prompt.text)
        for command in prompt.commands:
            self._print(f'    {command}')
        if prompt.details:
            self._print(prompt.details)
        default = _default_answer(prompt)

        if prompt.kind == INFO:
            return None

        if prompt.kind == CONFIRM:
            hint = '[Y/n]' if default else '[y/N]'
            answer = self._input(f'{hint}: ').strip().lower()
            return default if not answer else answer in ('y', 'yes')

        if prompt.kind == TEXT:
            if prompt.extra.get('secret'):
                import getpass
                self._print(f'{prompt.title}:')
                return getpass.getpass('> ').strip() or default
            answer = self._input(f'[{default}]: ').strip()
            return answer or default

        for i, label in enumerate(prompt.choices, 1):
            mark = ' (default)' if label == default else ''
            self._print(f'  {i}) {label}{mark}')
        while True:
            answer = self._input('Choose a number, or "q" to cancel: ').strip()
            if not answer:
                return default
            if answer.lower() == 'q':
                return None
            if answer.isdigit() and 1 <= int(answer) <= len(prompt.choices):
                return prompt.choices[int(answer) - 1]


class QtHandler(PromptHandler):
    """Shows Qt dialogs. Safe to call from any thread.

    Create it in the GUI thread. Calls from other threads are queued to the
    GUI thread and block until the user answers.
    """

    def __init__(self, parent=None):
        from qtpy.QtCore import QObject, Signal

        class _Bridge(QObject):
            request = Signal(object)

        self.parent = parent
        self._bridge = _Bridge(parent)
        self._bridge.request.connect(self._show)

    def ask(self, prompt: Prompt):
        from qtpy.QtCore import QThread

        if QThread.currentThread() == self._bridge.thread():
            return self._exec_dialog(prompt)
        holder, event = [], threading.Event()
        self._bridge.request.emit((prompt, holder, event))
        event.wait()
        return holder[0] if holder else None

    def _show(self, request):
        prompt, holder, event = request
        try:
            holder.append(self._exec_dialog(prompt))
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            event.set()

    # Choices with at most this many options are shown as buttons
    MAX_BUTTON_CHOICES = 3

    @staticmethod
    def _html(text):
        from cellacdc import html_utils
        return html_utils.paragraph((text or '').replace('\n', '<br>'))

    def _exec_dialog(self, prompt: Prompt):
        if prompt.kind == CHOICE and prompt.dialog == 'pytorch':
            return self._ask_pytorch(prompt)
        if prompt.kind == CHOICE:
            if len(prompt.choices or []) <= self.MAX_BUTTON_CHOICES:
                return self._ask_choice_buttons(prompt)
            return self._ask_choice_combobox(prompt)
        if prompt.kind == TEXT:
            return self._ask_text(prompt)
        return self._ask_message(prompt)

    def _ask_pytorch(self, prompt: Prompt):
        from cellacdc import apps
        win = apps.InstallPyTorchDialog(
            parent=self.parent,
            platforms=prompt.extra.get('platforms'),
            default_platform=prompt.default,
            target_python=prompt.extra.get('python'),
        )
        win.exec_()
        return None if win.cancel else win.platform

    def _ask_choice_buttons(self, prompt: Prompt):
        from cellacdc import widgets
        msg = widgets.myMessageBox(wrapText=False)
        buttons = msg.question(
            self.parent, prompt.title, self._html(prompt.text),
            buttonsTexts=(prompt.cancel_text, *prompt.choices),
            detailsText=prompt.details, commands=prompt.commands or None,
        )
        if msg.cancel:
            return None
        clicked = getattr(msg, 'clickedButton', None)
        for label, button in zip(prompt.choices, buttons[1:]):
            if button is clicked:
                return label
        return None

    def _ask_choice_combobox(self, prompt: Prompt):
        from cellacdc import apps
        win = apps.QDialogCombobox(
            prompt.title, list(prompt.choices), self._html(prompt.text),
            CbLabel='', parent=self.parent,
            defaultChannelName=prompt.default,
        )
        win.exec_()
        return None if win.cancel else win.selectedItemText

    def _ask_text(self, prompt: Prompt):
        from qtpy.QtWidgets import QLineEdit
        from cellacdc import apps
        win = apps.QLineEditDialog(
            title=prompt.title, msg=self._html(prompt.text),
            defaultTxt=prompt.default or '', parent=self.parent,
            allowText=True, allowEmpty=False,
        )
        if prompt.extra.get('secret'):
            win.entryWidget.setEchoMode(QLineEdit.EchoMode.Password)
        win.exec_()
        return None if win.cancel else win.entryWidget.text()

    def _ask_message(self, prompt: Prompt):
        from cellacdc import widgets
        msg = widgets.myMessageBox(wrapText=False)
        kwargs = dict(
            detailsText=prompt.details, commands=prompt.commands or None
        )
        text = self._html(prompt.text)
        if prompt.kind == INFO:
            msg.information(
                self.parent, prompt.title, text,
                buttonsTexts=(prompt.ok_text,), **kwargs
            )
            return None
        buttons = msg.question(
            self.parent, prompt.title, text,
            buttonsTexts=(prompt.cancel_text, prompt.ok_text), **kwargs
        )
        return not msg.cancel and msg.clickedButton is buttons[-1]
