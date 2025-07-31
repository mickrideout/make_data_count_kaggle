Implement a python script that uses the pyzoter library. The script will take 3 arguments: 
    - Zotero user id
    - Zotero api key
    - Zotero collection name


The should iterate through all papers in the zotero collection and for each do:
    - Only process items that are of type "journalArticle", "preprint", "conferencePaper", "webpage", "booksection"
    - Skip the item if it has a "CODE Repository" entry
    - Use the title for the paper and perform a google search for "{paper title} implements"
    - If there are any github repositorys, add it to the zotero item as a CODE Repository entry.