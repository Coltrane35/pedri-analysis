# io_compat.py
import sys

def setup_stdout_utf8() -> None:
    """
    Ustawia bezpieczne kodowanie stdout/stderr na UTF-8 i zastępuje
    niedrukowalne znaki, zamiast rzucać UnicodeEncodeError (Windows/CP1250).
    """
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # Nie panikujemy, jeśli środowisko nie wspiera reconfigure
        pass

def print_safe(*args, **kwargs):
    """
    print z gwarancją, że nie wywali się na problematycznych znakach.
    """
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        try:
            # Spróbuj wydrukować z backslashreplace
            sys.stdout.write(text.encode("utf-8", "backslashreplace").decode("utf-8"))
            end = kwargs.get("end", "\n")
            sys.stdout.write(end)
        except Exception:
            # Ostateczny fallback
            sys.stdout.write(text.encode("ascii", "ignore").decode("ascii") + "\n")
