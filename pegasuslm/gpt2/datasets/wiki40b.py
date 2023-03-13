from enum import Enum
class SpecialTokens(str, Enum):
    start_article = "_START_ARTICLE_"
    start_section = "_START_SECTION_"
    start_paragraph = "_START_PARAGRAPH_"
    newline = "_NEWLINE_"

# [e.value for e in SpecialTokens]

def get_paragraphs(text: str):
    paragraphs = []
    start_paragraph = False
    for line in text.splitlines():
        if line == "_START_PARAGRAPH_":
            start_paragraph = True
        elif start_paragraph:
            start_paragraph = False
            paragraphs.append(line.replace("_NEWLINE_", "\n"))
    return paragraphs

def preprocess(text: str):
    return "\n\n".join(get_paragraphs(text))
