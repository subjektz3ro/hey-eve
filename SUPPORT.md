# Support

Eve is a small, independently maintained hardware project. Help is provided on
a best-effort basis; there is no guaranteed response time.

Before opening an issue, read the setup instructions in [README.md](README.md)
and [deploy/README.md](deploy/README.md), then search existing issues. The
primary supported setup is 64-bit Raspberry Pi OS on a Raspberry Pi 5.
CPython 3.11, 3.12, and 3.13 are supported on compatible 64-bit glibc Linux
systems; managed service installation also requires systemd 243 or newer.
Alpine/musl, 32-bit operating systems, and Python 3.14 are not currently
supported.

Use the issue form that fits the request:

- **Bug report** for behavior that contradicts the documentation or a stable
  interface.
- **Setup or usage help** for installation, hardware, or configuration
  questions.
- **Feature request** for a new capability or behavior.

Include the operating system, CPU architecture, Python version, install method,
relevant hardware, the command or step that failed, and sanitized output.
Never post an API key, transcript, audio recording, remembered fact, username,
hostname, or personal path. Replace private values with descriptive
placeholders rather than partially masking them.

Security vulnerabilities do not belong in public issues. Follow
[SECURITY.md](SECURITY.md) and use the repository's private security-advisory
form.
