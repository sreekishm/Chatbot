import streamlit as st
from extract_keywords import extract_keywords
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
st.write("The system will answer questions from the passage")
model_id= "meta-llama/Llama-3.2-1B-Instruct"
keywords=st.session_state['keyword']
sentences=st.session_state['sentence']
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
question = st.chat_input("")
if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    q_keyword=extract_keywords(question)
    matched=[]
    for word in q_keyword:
        for k,values in keywords.items():
            if word.lower()==k.lower():
                matched.append(values)
    if not matched:
        with st.chat_message('assistant'):
                st.write("Answer not Found")
        st.session_state.messages.append({
                "role": "assistant",
                "content": "Answer not Found"
            })
        st.stop()
        
    matched=list(set(np.concatenate(matched).tolist()))
    result=[]
    for i in matched:
        result.append(sentences[i])
    messages = [
    {
        "role": "system",
        "content": (
            "You are a precise question answering system.\n\n"

            "Answer the question using ONLY information explicitly stated "
            "in the provided passage.\n\n"

            "GENERAL RULES:\n"
            "1. Identify exactly what the question is asking for before answering.\n"
            "2. Extract the smallest amount of text needed to answer the question.\n"
            "3. Do not repeat the subject of the question when it asks for "
            "information about that subject.\n"
            "4. Do not add information that is not explicitly stated.\n"
            "5. Do not make assumptions or generalizations.\n"
            "6. Preserve the exact meaning, number, quantity, and scope stated "
            "in the passage.\n"
            "7. Distinguish between singular and plural entities.\n"
            "8. Resolve pronouns such as he, she, it, this, these, they, and "
            "those using the surrounding passage.\n\n"
            "9. If the question is asking for an entity, return the full name of the entity"

            "WHO-IS QUESTIONS:\n"
            "If the question is 'Who is X?' or 'Who was X?', interpret it as "
            "'What is X's occupation or identifying role?'.\n"
            "Use the same extraction method used for an occupation question.\n"
            "Do NOT return the name of X.\n"
            "Return the occupation or identifying role explicitly stated in the passage.\n"
            "If an occupation is not explicitly stated, return another explicitly "
            "stated identifying role, title, nationality, or position.\n"
            "Do not add the person's name to the answer.\n\n"

            "EVENT QUESTIONS:\n"
            "If the question asks what happened to an entity, identify the "
            "specific event involving that entity.\n"
            "Do not apply an event involving one entity to other entities.\n"
            "For example, if the passage says 'one ship sank', the answer "
            "must refer to one ship, not all ships.\n\n"

            "RELATIONSHIP ANSWERS:\n"
            "When a question asks who, whom, what, where, or which entity is "
            "involved in a relationship or action, identify the entity or phrase "
            "that fills the requested role in the passage and return its complete "
            "answer span.\n"
            "Use the full phrase as it appears in the passage rather than "
            "extracting only the most obvious noun or proper noun from it.\n"

            "MULTIPLE ANSWERS:\n"
            "If multiple distinct entities satisfy the question, return all "
            "of them separated by 'and'.\n\n"

            "OUTPUT:\n"
            "Return only the answer. Do not explain your reasoning."    
            
        )
    },
    {
        "role": "user",
        "content": f"Text: {result}\n\nQuestion: {question}"
    }]
    inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt")
    inputs_ids= inputs["input_ids"].to(model.device)
    input_length = inputs_ids.shape[-1]
    outputs = model.generate(
    inputs_ids,
    max_new_tokens=100,
    do_sample=False)
    answer = tokenizer.decode(
    outputs[0][input_length:],
    skip_special_tokens=True
)
    with st.chat_message('assistant'):
        st.write(answer)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })