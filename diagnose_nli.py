"""Quick check of how the NLI model's labels come out on this machine."""
import numpy as np
from sentence_transformers import CrossEncoder

m = CrossEncoder("cross-encoder/nli-deberta-v3-base")
cfg = getattr(m, "config", None) or m.model.config
print("id2label:", cfg.id2label)
pairs = [
    ("The tower is 330 metres tall.", "The tower is 330 metres tall."),
    ("The tower is made of wrought iron.", "The tower is made of steel."),
    ("A man is eating pizza.", "A man is sleeping."),
    ("The Eiffel Tower is in Paris. It was completed in 1889.", "The Eiffel Tower was completed in 1889."),
]
out = m.predict(pairs, show_progress_bar=False)
print("raw output type:", type(out), "shape:", np.shape(out))
for (p, h), row in zip(pairs, np.atleast_2d(out)):
    print(np.round(row, 3), "|", h)
