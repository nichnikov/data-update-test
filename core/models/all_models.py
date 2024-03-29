# import os
#
# import torch
# from sentence_transformers import SentenceTransformer
# from transformers import T5Tokenizer, T5ForConditionalGeneration
#
# from core.settings import MODELS_DIR
#
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
# sbert_model = SentenceTransformer(str(os.path.join(MODELS_DIR, "all_sys_paraphrase.transformers")), device=DEVICE)
# t5_model = T5ForConditionalGeneration.from_pretrained(str(os.path.join(MODELS_DIR, "models_bss"))).to(DEVICE)
# t5_tokenizer = T5Tokenizer.from_pretrained(str(os.path.join(MODELS_DIR, "ruT5-large")))
#
# models = {"sbert-model": sbert_model, "t5-model": t5_model, "t5-tokenizer": t5_tokenizer}
