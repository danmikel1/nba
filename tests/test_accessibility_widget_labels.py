import re
import pathlib


def test_no_empty_streamlit_labels():
    """Fail if any Streamlit widget is invoked with an empty string label.

    This prevents the accessibility warning and future Streamlit errors.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    pattern = re.compile(r"st\.[a-zA-Z_]+\(\s*\"\"\s*(,|\))")

    matches = []
    for p in repo_root.rglob('*.py'):
        # skip virtualenv, test helpers, legacy folders and bytecode
        if ('venv' in p.parts
            or p.match('*/.venv/*')
            or '__pycache__' in p.parts
            or any(part.startswith('g(old') for part in p.parts)
            or p.match('tests/**')):
            continue
        text = p.read_text(encoding='utf-8')
        for m in pattern.finditer(text):
            # record file:line for easier debugging
            lineno = text.count('\n', 0, m.start()) + 1
            matches.append(f"{p.relative_to(repo_root)}:{lineno}")

    assert not matches, f"Found Streamlit calls with empty labels (fix by providing a non-empty label and use label_visibility='collapsed'): {matches}"
