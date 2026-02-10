import re

def extract_strategic_objectives(text: str):
    # Matches SO1.1: ... (simple)
    pattern = r"(SO\d+\.\d+)\s*:\s*(.+)"
    return [{"id": m.group(1), "type": "strategy", "text": m.group(2).strip()} 
            for m in re.finditer(pattern, text)]

def extract_actions(text: str):
    # Capture:
    # Action A1.1: Establish Curriculum Innovation Task Force
    # <body until next Action...>
    action_pat = r"Action\s+(A\d+\.\d+)\s*:\s*(.+?)(?=\n\nAction|\Z)"
    linked_pat = r"Linked to:\s*([SO\d\.,\s]+)"

    actions = []
    for m in re.finditer(action_pat, text, flags=re.S):
        aid = m.group(1).strip()
        desc = m.group(2).strip()   # this contains title + body (everything after ':')

        # Title = first line of desc
        lines = [ln.strip() for ln in desc.splitlines() if ln.strip()]
        title = lines[0] if lines else ""
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

        # Find linked_to inside desc/body
        lm = re.search(linked_pat, desc)
        linked = []
        if lm:
            linked = [x.strip() for x in lm.group(1).split(",") if x.strip()]

        # Build embedding text: include real title
        full_text = f"Action {aid}: {title}\n{desc}".strip()

        actions.append({
            "id": aid,
            "type": "action",
            "title": title,       # ✅ new field (headline)
            "text": full_text,
            "linked_to": linked,
        })

    return actions