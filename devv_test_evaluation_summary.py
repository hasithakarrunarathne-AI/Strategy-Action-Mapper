# dev_test_expert_summary.py
from src.evaluation import summarize_expert_reviews
import json

summary = summarize_expert_reviews("outputs/expert_review.csv")
print(json.dumps(summary, indent=2))
