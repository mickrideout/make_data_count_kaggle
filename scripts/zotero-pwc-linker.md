
Implement a python script that uses the pyzotero library. The script will take 3 arguments:
    - Path to the paperswithcode links json file 
    - Zotero user id
    - Zotero api key
    - Zotero collection name

A same of the linker json is given below in the Sample Jason section.

The should iterate through all papers in the zotero collection and for each do:
    - Only process items that are of type "journalArticle", "preprint", "conferencePaper", "webpage", "booksection"
    - Use the title for the paper, iterate through each paper_title in the linker json. Perform a fuzzy match on paper_title
    - If the title matches, add the "repo_url" from the json as a Web Link attachment to the zotero paper.




## Sample Json
[
  {
    "paper_url": "https://paperswithcode.com/paper/odyssey-a-public-gpu-based-code-for-general",
    "paper_title": "Odyssey: A Public GPU-Based Code for General-Relativistic Radiative Transfer in Kerr Spacetime",
    "paper_arxiv_id": "1601.02063",
    "paper_url_abs": "https://arxiv.org/abs/1601.02063v2",
    "paper_url_pdf": "https://arxiv.org/pdf/1601.02063v2.pdf",
    "repo_url": "https://github.com/LeonGeiger/Kerr",
    "is_official": false,
    "mentioned_in_paper": false,
    "mentioned_in_github": true,
    "framework": "none"
  },
  {
    "paper_url": "https://paperswithcode.com/paper/mapping-natural-language-instructions-to",
    "paper_title": "Mapping Natural Language Instructions to Mobile UI Action Sequences",
    "paper_arxiv_id": "2005.03776",
    "paper_url_abs": "https://arxiv.org/abs/2005.03776v2",
    "paper_url_pdf": "https://arxiv.org/pdf/2005.03776v2.pdf",
    "repo_url": "https://github.com/deepneuralmachine/seq2act-tensorflow",
    "is_official": false,
    "mentioned_in_paper": false,
    "mentioned_in_github": false,
    "framework": "tf"
  },
