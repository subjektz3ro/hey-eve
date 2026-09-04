#!/usr/bin/env python3
"""Render the path-neutral systemd template into a unit for THIS machine.

`deploy/eve@.service` is a template, not a file to copy. It used to be a
literal unit naming one account and one absolute checkout path, so the
repository could only run from one person's home directory — and that fact
was tracked in git, where scripts/check_release_hygiene.py now refuses it.

Every host-specific value is resolved here instead: the checkout, the exact
`uv` that install.sh used, the selected configuration and cache directories,
and the checkout's virtual environment.

The output carries a digest of (template + this renderer) on its first line.
deploy/ship.sh compares that digest against the commit it is about to deploy,
because installing a unit is root-equivalent and therefore deliberately
outside the deploy account's narrow stop/start permission. If the contract
changed, ship.sh stops and tells the operator to rerun install.sh instead of
restarting a service under a stale unit.

    uv run python deploy/render_service.py \
        --template deploy/eve@.service --checkout "$PWD" --uv "$(command -v uv)" \
        --config-dir ~/.config/eve --uv-cache-dir "$(uv cache dir)" \
        --supplementary-groups audio video tty bluetooth input \
        --output /tmp/eve@.service
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

RENDERER = Path(__file__).resolve()
RELATED_CONTRACT_FILES = (
    "eve-speaker@.service",
    "eve-speaker@.timer",
)

# Every placeholder the template may use. Rendering fails if one is left
# behind: a systemd unit with a literal @UV_EXECUTABLE@ in ExecStart starts
# and then dies on every restart, which is a worse failure than not starting.
PLACEHOLDERS = (
    "@WORKING_DIRECTORY@",
    "@UV_EXECUTABLE@",
    "@CONFIG_DIRECTORY@",
    "@UV_CACHE_DIRECTORY@",
    "@VENV_DIRECTORY@",
    "@SUPPLEMENTARY_GROUPS@",
)


def contract_digest(
    template: bytes,
    renderer: bytes,
    related: tuple[tuple[str, bytes], ...] = (),
) -> str:
    """Digest of the main template, renderer, and related installed units.

    Every part matters. A renderer that starts emitting a different
    ReadWritePaths= changes the installed unit just as surely as editing the
    template does, and a changed speaker service or timer likewise needs an
    interactive reinstall. The separators are NUL so blobs cannot be slid
    across boundaries into a different contract with the same digest.
    """
    stream = b"template\0" + template + b"\0renderer\0" + renderer
    for name, contents in related:
        stream += b"\0unit-file\0" + name.encode() + b"\0" + contents
    return hashlib.sha256(stream).hexdigest()


def render(
    template: str,
    *,
    checkout: Path,
    uv: Path,
    config_dir: Path,
    uv_cache_dir: Path,
    supplementary_groups: tuple[str, ...] = (),
) -> str:
    values = {
        "@WORKING_DIRECTORY@": str(checkout),
        "@UV_EXECUTABLE@": str(uv),
        "@CONFIG_DIRECTORY@": str(config_dir),
        "@UV_CACHE_DIRECTORY@": str(uv_cache_dir),
        "@VENV_DIRECTORY@": str(checkout / ".venv"),
    }
    for placeholder, value in values.items():
        # A systemd unit is line-oriented; a newline in a rendered path would
        # silently invent a directive rather than fail.
        if not value.startswith("/"):
            raise SystemExit(
                f"{placeholder} must be an absolute path, got {value!r}")
        if "\n" in value or "\r" in value:
            raise SystemExit(f"{placeholder} must be a single line: {value!r}")
        # These strings land verbatim in WorkingDirectory=, Environment= and
        # ExecStart=. Whitespace and quoting need systemd-specific escaping,
        # while '%' is a specifier even inside many quoted directives. Eve's
        # supported Linux layout needs none of those characters, so failing
        # closed is clearer than silently rendering a different path.
        if not re.fullmatch(r"/[A-Za-z0-9_./+-]*", value):
            raise SystemExit(
                f"{placeholder} contains characters that cannot be rendered "
                f"safely in a systemd unit: {value!r}"
            )
        template = template.replace(placeholder, value)

    invalid_groups = [
        group for group in supplementary_groups
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", group)
    ]
    if invalid_groups:
        raise SystemExit(
            "supplementary group names contain unsupported characters: "
            + ", ".join(repr(group) for group in invalid_groups)
        )
    template = template.replace(
        "@SUPPLEMENTARY_GROUPS@", " ".join(supplementary_groups)
    )

    # Scan for ANY @UPPER_CASE@ token, not just the ones this renderer knows.
    # Checking only the known list would let a placeholder added to the
    # template and forgotten here ship as a literal — a unit that installs
    # cleanly and then fails on every start. Lowercase is untouched, so the
    # `eve@someone` in the template's own comments is not a false positive.
    leftover = sorted(set(re.findall(r"@[A-Z_]+@", template)))
    if leftover:
        raise SystemExit(
            "template still contains unrendered placeholders: "
            + ", ".join(leftover)
        )
    return template


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--uv-cache-dir", type=Path, required=True)
    parser.add_argument(
        "--supplementary-groups",
        nargs="*",
        default=(),
        help="host groups that exist and should be added to the service user",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    template_bytes = args.template.read_bytes()
    related: tuple[tuple[str, bytes], ...] = ()
    if args.template.name == "eve@.service":
        related = tuple(
            (name, (args.template.parent / name).read_bytes())
            for name in RELATED_CONTRACT_FILES
        )
    digest = contract_digest(template_bytes, RENDERER.read_bytes(), related)
    # expanduser, but deliberately NOT resolve(). resolve() consults the
    # filesystem, so rendering a unit for a path that does not exist on THIS
    # machine — the normal case when checking the contract, and the whole
    # case in tests — rewrote it. On macOS it silently redirected a home
    # path through the firmlink under /System/Volumes/Data. The caller passes
    # absolute paths already; the check inside render() is what enforces it.
    checkout = args.checkout.expanduser()
    body = render(
        template_bytes.decode(),
        checkout=checkout,
        uv=args.uv.expanduser(),
        config_dir=args.config_dir.expanduser(),
        uv_cache_dir=args.uv_cache_dir.expanduser(),
        supplementary_groups=tuple(args.supplementary_groups),
    )
    # First line, so ship.sh can read it with a single `read -r` on the host
    # rather than parsing the unit.
    args.output.write_text(
        f"# eve-unit-contract-sha256={digest}\n"
        f"# eve-config-dir={args.config_dir.expanduser()}\n"
        f"# eve-checkout-dir={checkout}\n"
        f"{body}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
