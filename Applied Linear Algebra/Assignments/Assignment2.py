
import numpy as np


TAGS = [
    "research", "innovation", "education", "university", "students",
    "faculty", "campus", "engineering", "medicine", "technology",
    "curriculum", "collaboration", "publication", "laboratory",
    "scholarship", "mentorship", "internship", "entrepreneurship",
    "accreditation", "alumni"
]


def get_word_vector(model, tags):
    dropped_words = []
    word_vectors = []

    for tag in tags:
        if tag in model.key_to_index:
            word_vectors.append(model[tag])
        else:
            dropped_words.append(tag)

    return dropped_words, word_vectors


def build_tag_matrix(model, tags):
    if len(tags) != 20:
        raise ValueError("Exactly 20 tags are required")

    dropped_words, word_vectors = get_word_vector(model, tags)

    if len(dropped_words) > 0:
        raise ValueError("Some tags are missing: " + str(dropped_words))

    T = np.array(word_vectors)

    return T
