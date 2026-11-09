# Reference libraries

Download the two Enrichr gene-set libraries as plain text and save them in this folder with these names:

| File | Enrichr library |
|---|---|
| `KEGG_2021_Human.txt` | KEGG_2021_Human |
| `Reactome_2022.txt` | Reactome_2022 |

Download links (Enrichr "Libraries" page, https://maayanlab.cloud/Enrichr/#libraries):

- https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=KEGG_2021_Human
- https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=Reactome_2022

The files are not included in this repository because KEGG content is subject to KEGG's licensing terms.
After filtering to sets with at least eight genes, the checker uses 314 KEGG and 1,593 Reactome sets
covering 12,120 unique gene symbols.
