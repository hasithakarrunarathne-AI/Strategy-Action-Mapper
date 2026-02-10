import numpy as np

def cosine(a, b):
    a = np.array(a); b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

def label(score: float):
    if score >= 0.80: return "Strong"
    if score >= 0.55: return "Partial"
    return "Weak"