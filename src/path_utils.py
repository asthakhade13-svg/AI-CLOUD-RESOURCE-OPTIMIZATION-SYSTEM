import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve_path(rel_path: str) -> str:
    """
    Resolves a relative file path (e.g., 'artifacts/scaler.pkl', 'data/cleaned_workload.csv')
    by checking candidate directories in priority order:
    1. Exact path if already exists (handles absolute or valid relative paths)
    2. PROJECT_ROOT / rel_path
    3. CWD / rel_path
    4. /app / rel_path (Docker container root)
    5. Parent directory of CWD / rel_path

    Returns the first matching absolute path, or falls back to PROJECT_ROOT / rel_path.
    """
    if not rel_path:
        return rel_path

    candidates = [
        rel_path,
        os.path.join(PROJECT_ROOT, rel_path),
        os.path.join(os.getcwd(), rel_path),
        os.path.join("/app", rel_path) if os.path.exists("/app") else None,
        os.path.join(os.path.dirname(os.getcwd()), rel_path),
    ]

    for cand in candidates:
        if cand and os.path.exists(cand):
            return os.path.abspath(cand)

    # Fallback to PROJECT_ROOT / rel_path
    return os.path.abspath(os.path.join(PROJECT_ROOT, rel_path))
