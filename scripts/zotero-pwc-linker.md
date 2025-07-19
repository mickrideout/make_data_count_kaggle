
Implement a python script that uses the zotero web api. The script will take 3 arguments:
    - Path to the papers-with-code links json file
    - Zotero user id
    - Zotero api key
    - Zotero collection name

A same of the linker json is given below in the Sample Jason section.

The should iterate through all papers in the zotero collection and for each do:
    - Use the title for the paper, iterate through each paper_title in the linker json. Perform a fuzzy match on paper_title
    - If the title matches, add the "paper_url" from the json as a Web Link attachment to the zotero paper.




## Sample Json
[
  {
    "paper_url": "https://paperswithcode.com/paper/dynamic-network-model-from-partial",
    "arxiv_id": "1805.10616",
    "nips_id": null,
    "openreview_id": null,
    "title": "Dynamic Network Model from Partial Observations",
    "abstract": "Can evolving networks be inferred and modeled without directly observing\ntheir nodes and edges? In many applications, the edges of a dynamic network\nmight not be observed, but one can observe the dynamics of stochastic cascading\nprocesses (e.g., information diffusion, virus propagation) occurring over the\nunobserved network. While there have been efforts to infer networks based on\nsuch data, providing a generative probabilistic model that is able to identify\nthe underlying time-varying network remains an open question. Here we consider\nthe problem of inferring generative dynamic network models based on network\ncascade diffusion data. We propose a novel framework for providing a\nnon-parametric dynamic network model--based on a mixture of coupled\nhierarchical Dirichlet processes-- based on data capturing cascade node\ninfection times. Our approach allows us to infer the evolving community\nstructure in networks and to obtain an explicit predictive distribution over\nthe edges of the underlying network--including those that were not involved in\ntransmission of any cascade, or are likely to appear in the future. We show the\neffectiveness of our approach using extensive experiments on synthetic as well\nas real-world networks.",
    "short_abstract": null,
    "url_abs": "http://arxiv.org/abs/1805.10616v4",
    "url_pdf": "http://arxiv.org/pdf/1805.10616v4.pdf",
    "proceeding": "NeurIPS 2018 12",
    "authors": [
      "Elahe Ghalebi",
      "Baharan Mirzasoleiman",
      "Radu Grosu",
      "Jure Leskovec"
    ],
    "tasks": [
      "model",
      "Open-Ended Question Answering"
    ],
    "date": "2018-05-27",
    "conference_url_abs": "http://papers.nips.cc/paper/8192-dynamic-network-model-from-partial-observations",
    "conference_url_pdf": "http://papers.nips.cc/paper/8192-dynamic-network-model-from-partial-observations.pdf",
    "conference": "dynamic-network-model-from-partial-1",
    "reproduces_paper": null,
    "methods": [
      {
        "name": "ooJpiued",
        "full_name": "ooJpiued",
        "description": "Please enter a description about the method here",
        "introduced_year": 2000,
        "source_url": "http://arxiv.org/abs/1805.10616v4",
        "source_title": "Dynamic Network Model from Partial Observations",
        "code_snippet_url": null,
        "main_collection": {
          "name": "Language Models",
          "description": "**Language Models** are models for predicting the next word or character in a document. Below you can find a continuously updating list of language models.\r\n\r\n",
          "parent": null,
          "area": "Natural Language Processing"
        }
      }
    ]
  },
