# AI Hands-on Assignement 2

This exercise is an extension of the first AI hands-on project, repo [here](https://github.com/nichaos2/ai-hand-on-hw1) 

# Problem Description

Implement Retrieval-Augmented Generation (RAG) on the documents related to HW1.

# Knowledge Base & RAG system

## 1.1 Document Collection

Sources:
- https://en.wikipedia.org/wiki/Elo_rating_system
- https://en.wikipedia.org/wiki/Encyclopaedia_of_Chess_Openings
- https://en.wikipedia.org/wiki/Time_control
- https://en.wikipedia.org/wiki/Chess_theory
- https://en.wikipedia.org/wiki/Chess_tactic

The sources are only from wikipedia and describe various chess concepts in detail.

_Note_: I have used this [tool](https://wikitext.eluni.co/) to turn wikipedia articles to txt files.

## 1.2 Vector Store

During the course lab we have compared 3 chunking strategies; fixed-size, recursive and semantic.

_Disclaimer_: Time limit is pressing so for the time being we will choose one and then if we have time we might be able to compare in the exercise

- We will use the recursive chunking in this exercise
    - `RecursiveCharacterTextSplitter` handles empty lines beautifully by splitting on \n\n first.
    - `chunk_size=1000` ensures chunks are large enough to hold a meaningful thought.
    - `chunk_overlap=200` prevents sentences from being awkwardly cut in half between chunks.
    - _Note_: documents have many empty lines between paragraphs, maximum is 45712 chars and min is 2634 chars
- We utilize Chroma (via langchain-chroma) for its specific implementation features as show in the lab (mainly becuase it runs in local env)
- For the embedding we will use Gemini


## 1.3 Retrieval 

- code to retrieve in `rag.py`
- `run_rag_retrieve.py` has a simple snippet to test the `retrieval` snippet