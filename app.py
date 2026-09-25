import re
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

st.set_page_config(page_title="Nepali-Magar Translator", page_icon="🇳🇵")
st.title("Nepali ⇄ Magar Machine Translation")
st.write("Transformer-Based Magar-to-Nepali Translation System.")

@st.cache_resource
def load_model():
    BASE_MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
    
    tokenizer = AutoTokenizer.from_pretrained(".")
    
    # Low CPU memory usage logic
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_NAME, 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    vocab_size = max(len(tokenizer), 250055)
    base_model.resize_token_embeddings(vocab_size)
    
    model = PeftModel.from_pretrained(base_model, ".")
    model.eval()
    return tokenizer, model

try:
    with st.spinner("Loading Model... (यसले १-२ मिनेट लिन सक्छ)"):
        tokenizer, model = load_model()

    input_text = st.text_area("यहाँ नेपाली वा मगर भाषाको वाक्य लेख्नुहोस्:", height=120)

    if st.button("Translate (अनुवाद गर्नुहोस्)"):
        if input_text.strip():
            tokenizer.src_lang = "hi_IN"
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            forced_bos_id = tokenizer.lang_code_to_id.get("hi_IN", None)
            
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs, max_new_tokens=128, num_beams=4, early_stopping=True, forced_bos_token_id=forced_bos_id
                )
            
            raw_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            devanagari_only = re.sub(r'[^\u0900-\u097F0-9\s.,!?\'"-]', '', raw_output)
            cleaned_output = re.sub(r'\s+', ' ', devanagari_only).strip()
            
            st.success("Translated Output:")
            st.write(cleaned_output if cleaned_output else raw_output)
            
except Exception as e:
    st.error(f"Error loading model: {e}")
