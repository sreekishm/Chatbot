import streamlit as st
from prepare_data import prepare_data
def main_page():

    text = st.text_area(
        "Enter your passage",
        height=300,
        placeholder="Enter or paste your passage here"
    )

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        extract_button = st.button(
            "Learn",
            use_container_width=True
        )

    if extract_button:

        if text.strip() == "":
            st.warning("Please Enter some text")

        else:
            keyword, sentence = prepare_data(text)

            st.session_state["keyword"] = keyword
            st.session_state["sentence"] = sentence

            st.switch_page("answer_question.py")


main = st.Page(main_page, title="Extract Relation")
answer = st.Page("answer_question.py", title="Answer Question")

pg = st.navigation([main, answer],position='hidden')

pg.run()