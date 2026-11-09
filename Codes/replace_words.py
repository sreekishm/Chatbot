import spacy

nlp = spacy.load("en_core_web_sm")


PERSON_PRONOUNS = {
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themselves"
}


def get_complete_name(token):

    name_tokens = [token]

    for child in token.children:

        if child.dep_ in ("compound", "flat", "flat:name"):
            name_tokens.append(child)

    name_tokens.sort(key=lambda x: x.i)

    return " ".join(
        token.text for token in name_tokens
    )


def get_subject_people(sent):
    people = []

    for token in sent:

        if token.dep_ in ("nsubj", "nsubjpass"):

            if token.pos_ == "PROPN":

                name = get_complete_name(token)

                if name not in people:
                    people.append(name)

            else:
                for ent in sent.ents:

                    if ent.start <= token.i < ent.end:

                        if ent.text not in people:
                            people.append(ent.text)

    return people


def get_candidate_people(processed_sentences, current_sentence):
    candidates = []
    for sentence_info in processed_sentences:

        for person in sentence_info["subject_people"]:

            if person not in candidates:
                candidates.append(person)

    current_people = get_subject_people(
        current_sentence
    )

    for person in current_people:

        if person not in candidates:
            candidates.append(person)

    return candidates


def replace_words(text):

    doc = nlp(text)

    processed_sentences = []

    for sent in doc.sents:
        candidates = get_candidate_people(
            processed_sentences,
            sent
        )

        words = []

        for token in sent:

            word = token.text
            lower = word.lower()

            if lower in PERSON_PRONOUNS:

                if candidates:
                    person = candidates[-1]

                    if lower in {
                        "he",
                        "she",
                        "they"
                    }:

                        word = person

                    # ----------------------------
                    # Object pronouns
                    # ----------------------------

                    elif lower in {
                        "him",
                        "her",
                        "them"
                    }:

                        word = person

                    # ----------------------------
                    # Possessive pronouns
                    # ----------------------------

                    elif lower in {
                        "his",
                        "hers",
                        "their",
                        "theirs"
                    }:

                        word = person + "'s"

                    # ----------------------------
                    # Reflexive pronouns
                    # ----------------------------

                    elif lower in {
                        "himself",
                        "herself",
                        "themselves"
                    }:

                        word = person

            words.append(word)

        new_sentence = ""

        for i, word in enumerate(words):

            if i == 0:

                new_sentence += word

            elif word in {
                ".",
                ",",
                "!",
                "?",
                ";",
                ":"
            }:

                new_sentence += word

            else:

                new_sentence += " " + word

        current_subject_people = get_subject_people(
            sent
        )

        processed_sentences.append({
            "original": sent.text,
            "rewritten": new_sentence,
            "subject_people": current_subject_people
        })

    return " ".join(
        item["rewritten"]
        for item in processed_sentences
    )