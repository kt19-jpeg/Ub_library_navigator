# Citation Tool (Layer 2, Slice 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deterministic APA/MLA/Chicago citation formatting rendered inline (collapsed by
default) with every result the agent shows — no conversational "cite that" fast path.

**Architecture:** Real bibliographic data (journal title, volume, issue, pages, DOI, ISBN,
publisher, author list) already arrives in OneSearch's existing brief-search response
(`pnx.addata`) but isn't parsed today. New optional `SearchCandidate` fields capture it; a new,
pure `agent/citation_tool.py` formats it per style; `compose.py` attaches a citation per shown
result to its output; the CLI and Streamlit UI each render that their own way.

**Tech Stack:** Same as prior slices. No new packages.

**Spec:** `docs/superpowers/specs/2026-10-04-citation-tool-design.md`

## Global Constraints

- `format_citation()` is pure and deterministic — no LLM call, matching the original brief's "a
  deterministic tool."
- New `SearchCandidate` fields are added with defaults (`= None`), never replacing the existing
  `author` field — nothing that reads it today may break.
- `compose.py` stays UI-agnostic: it attaches citation data to its output; the CLI and Streamlit
  UI each decide how to display it.
- A missing field is omitted cleanly from a citation — it must never render as a literal `"None"`
  or leave a dangling comma with no closing period. Two real bugs of exactly this kind were found
  and fixed live during this plan's verification (Task 2); their regression tests are part of the
  plan, not optional extras.
- Documented simplification, not full style-guide fidelity: 1-2 authors listed in full, 3+ as
  "first author et al.", for every style.
- Commit messages are `type: short description`. **Never add a Co-Authored-By or "Generated with
  Claude Code" line.** Commit after each task; push once after the whole slice is reviewed.

## File Structure

| File | Responsibility |
|---|---|
| `onesearch/models.py` (modify) | New optional `SearchCandidate` citation fields |
| `onesearch/parser.py` (modify) | Parse those fields from real `pnx.addata` |
| `agent/citation_tool.py` | `format_citation()` — APA/MLA/Chicago, pure functions |
| `agent/compose.py` (modify) | `build_citations()`, attached to `compose()`'s output |
| `agent/models.py` (modify) | `AgentState.citations` |
| `agent/cli.py` (modify) | `--citations` flag + rendering |
| `streamlit_app/Home.py` (modify) | `st.expander` per shown result |
| `tests/fixtures/onesearch_article_doc.json` | Real captured journal-article document |
| `tests/fixtures/onesearch_book_doc.json` | Real captured book document |

---

### Task 1: Citation fields on `SearchCandidate` + parser

Real UB data confirmed live 2026-10-04: a peer-reviewed article's `pnx.addata` has
`au`/`addau` (author list), `jtitle`, `volume`, `issue`, `spage`/`epage`/`pages`, `doi`, `pub`; a
book's has `addau`, `btitle`, `isbn`, `pub`, `cop`. All of it is already present in the existing
brief search response — no extra network call. One real field-naming surprise: which key holds
the author list is not clean by record type — a real article had `au` with `addau` absent; a real
book had `addau` with `au` absent. The parser checks both.

**Files:**
- Create: `tests/fixtures/onesearch_article_doc.json`, `tests/fixtures/onesearch_book_doc.json`
  (full real captured documents, not just their `addata`)
- Modify: `onesearch/models.py`, `onesearch/parser.py`
- Test: `tests/test_parser.py` (append)

**Interfaces:**
- Produces: new `SearchCandidate` fields, all `| None = None` — `authors: list[str]`,
  `journal_title: str`, `volume: str`, `issue: str`, `pages: str`, `doi: str`, `isbn: str`,
  `publisher: str`, `place_of_publication: str`. `journal_title` is what later tells an article
  from a book-like record (Task 2).


- [ ] **Step 1: Save the real fixture documents**

Create `tests/fixtures/onesearch_article_doc.json`:

```json
{
  "pnx": {
    "links": {
      "openurl": [
        "$$Topenurl_article"
      ],
      "openurlfulltext": [
        "$$Topenurlfull_article"
      ],
      "backlink": [
        "$$Uhttps://www.ncbi.nlm.nih.gov/pubmed/34852489$$D View this record in MEDLINE/PubMed$$Hfree_for_read$$Pomit_proxy_true",
        "$$Uhttps://www.osti.gov/biblio/1832071$$D View this record in Osti.gov$$Hfree_for_read$$Pomit_proxy_true",
        "$$Uhttps://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-462360$$DView record from Swedish Publication Index (Uppsala universitet)$$Hfree_for_read$$Pomit_proxy_true"
      ],
      "thumbnail": [
        "$$Tsyndetics_thumb_exl"
      ],
      "linktohtml": [
        "$$Uhttps://gate.lib.buffalo.edu/login?&url=https://pubs.aip.org/jcp/article-lookup/doi/10.1063/5.0059356$$EHTML$$Gscitation$$H$$Pomit_proxy_true"
      ],
      "linkunpaywall": [
        "$$Uhttps://aip.scitation.org/doi/pdf/10.1063/5.0059356$$GUnpaywall$$Hfree_for_read"
      ],
      "linktorsrcadditional": [
        "$$Uhttps://aip.scitation.org/doi/pdf/10.1063/5.0059356$$EView_record_in_Unpaywall$$FView_record_in_$$GUnpaywall$$Hfree_for_read$$Pomit_proxy_true"
      ],
      "openurladditional": [
        "$$Topenurlfull_article"
      ]
    },
    "search": {
      "fulltext": [
        "true"
      ],
      "sourceid": [
        "ADTOC"
      ],
      "recordid": [
        "eNp9kstuEzEUhkcIRENhwQugEWwaRFpfZuwxC6SQlItUCVUCtpY94ySuZuzBl1R5Hx4UJ5MG2qKubNnf_5_j_zjLXkJwCgHBZ-UpACXDJXmUjSCo2IQSBh5nIwAQnDACyFH2zPsrAACkqHiaHeGiKlFRsVH2-zIKE2KXz1aq0z64TT6zXWdNPnd6rVwuTJPPRRBSeOXzk8vZ_ON4d3hfeG6W2qgtM-zG7_NpDLYTQSe_rUaboJztlRNStzpscpEqLfPadn0MO0y0eX0w7J1dOtH559mThWi9erFfj7Mfn86_z75MLr59_jqbXkwkhSxMqkIIioBkDS4pUlJS0hSSUlI3WDUSVxgumJAAl3XFCryQjZANQAgphIgoJT7O3g6-0fRicy3alvdOd8JtOAR8GzUv-T7qBL8bYH-t-igPpBWaz_XPKbduyWPkBUGYgIR_GPDEdqqplQlOtLdUt2-MXvGlXfOKwAIhlgxeDwbWB819rYOqV7U1RtWBwwojQGGCTvZVnP0VlQ88RVmrthVG2eg5IqAkANCd35s76JWNLg1gR4GC0IJViXr1b9uHfm9-UALOBqB21nunFjx1thtleoVu_xvc-I7ioZD3E_E3rgd4bd1fkPfN4iH4vvMfkQ8BYg"
      ],
      "title": [
        "Quantum Chemistry Common Driver and Databases (QCDB) and Quantum Chemistry Engine (QCEngine): Automation and interoperability among computational chemistry programs",
        "The Journal of chemical physics"
      ],
      "subject": [
        "Computational chemistry",
        "Physics",
        "Practice Guidelines as Topic",
        "Quantum chemistry",
        "Vibrational spectra"
      ],
      "recordtype": [
        "article"
      ],
      "creator": [
        "Smith, Daniel G. A.",
        "Lolinco, Annabelle T.",
        "Glick, Zachary L.",
        "Lee, Jiyoung",
        "Alenaizan, Asem",
        "Barnes, Taylor A.",
        "Borca, Carlos H.",
        "Di Remigio, Roberto",
        "Dotson, David L.",
        "Ehlert, Sebastian",
        "Heide, Alexander G.",
        "Herbst, Michael F.",
        "Hermann, Jan",
        "Hicks, Colton B.",
        "Horton, Joshua T.",
        "Hurtado, Adrian G.",
        "Kraus, Peter",
        "Kruse, Holger",
        "Lee, Sebastian J. R.",
        "Misiewicz, Jonathon P.",
        "Naden, Levi N.",
        "Ramezanghorbani, Farhad",
        "Scheurer, Maximilian",
        "Schriber, Jeffrey B.",
        "Simmonett, Andrew C.",
        "Steinmetzer, Johannes",
        "Wagner, Jeffrey R.",
        "Ward, Logan",
        "Welborn, Matthew",
        "Altarawy, Doaa",
        "Anwar, Jamshed",
        "Chodera, John D.",
        "Dreuw, Andreas",
        "Kulik, Heather J.",
        "Liu, Fang",
        "Mart\u00ednez, Todd J.",
        "Matthews, Devin A.",
        "Schaefer, Henry F.",
        "\u0160poner, Ji\u0159\u00ed",
        "Turney, Justin M.",
        "Wang, Lee-Ping",
        "De Silva, Nuwan",
        "King, Rollin A.",
        "Stanton, John F.",
        "Gordon, Mark S.",
        "Windus, Theresa L.",
        "Sherrill, C. David",
        "Burns, Lori A."
      ],
      "creationdate": [
        "2021"
      ],
      "orcidid": [
        "https://orcid.org/0000-0002-7545-4416",
        "https://orcid.org/0000-0003-1322-4997",
        "https://orcid.org/0000-0003-2371-1318",
        "https://orcid.org/0000-0003-2852-5864",
        "https://orcid.org/0000-0001-7006-9378",
        "https://orcid.org/0000-0003-3072-9946",
        "https://orcid.org/0000-0002-1686-778X",
        "https://orcid.org/0000-0002-5452-9239",
        "https://orcid.org/0000-0003-0378-7921",
        "https://orcid.org/0000-0002-0871-664X",
        "https://orcid.org/0000-0001-8626-0900",
        "https://orcid.org/0000-0003-0592-3464",
        "https://orcid.org/0000-0003-2834-318X",
        "https://orcid.org/0000-0002-5570-7666",
        "https://orcid.org/0000-0002-1173-4187",
        "https://orcid.org/0000-0002-5921-9272",
        "https://orcid.org/0000-0002-3692-5027",
        "https://orcid.org/0000-0003-0542-119X",
        "https://orcid.org/0000-0003-0252-2083",
        "https://orcid.org/0000-0001-6906-8191",
        "https://orcid.org/0000-0002-1443-5764",
        "https://orcid.org/0000-0001-5879-2942",
        "https://orcid.org/0000-0002-9780-8123",
        "https://orcid.org/0000-0002-6425-9551",
        "https://orcid.org/0000-0001-7809-771X",
        "https://orcid.org/0000-0002-7795-4422",
        "https://orcid.org/0000-0001-6065-3167",
        "https://orcid.org/0000-0003-2345-9781",
        "https://orcid.org/0000-0003-2493-6893",
        "https://orcid.org/0000-0002-2460-8059",
        "https://orcid.org/0000-0003-2795-5483",
        "https://orcid.org/0000-0002-0560-1513",
        "https://orcid.org/0000-0003-0900-2849",
        "https://orcid.org/0000-0002-5862-5113",
        "https://orcid.org/0000-0002-2779-0749",
        "https://orcid.org/0000-0001-8694-7200",
        "https://orcid.org/0000-0003-3659-0711",
        "https://orcid.org/0000-0002-1323-5939",
        "https://orcid.org/0000-0003-1721-0330",
        "https://orcid.org/0000-0001-6558-6186",
        "https://orcid.org/0000-0002-4359-5003",
        "https://orcid.org/0000-0001-6448-0873",
        "https://orcid.org/0000-0001-9396-1094",
        "https://orcid.org/0000-0002-4798-8947",
        "https://orcid.org/0000-0001-9342-0191",
        "https://orcid.org/0000-0003-0683-7613",
        "https://orcid.org/0000-0002-5216-2859",
        "https://orcid.org/0000000224608059",
        "https://orcid.org/0000000297808123",
        "https://orcid.org/0000000275454416",
        "https://orcid.org/0000000327955483",
        "https://orcid.org/0000000306837613",
        "https://orcid.org/0000000317210330",
        "https://orcid.org/0000000186260900",
        "https://orcid.org/0000000160653167",
        "https://orcid.org/0000000227790749",
        "https://orcid.org/000000032834318X",
        "https://orcid.org/0000000170069378",
        "https://orcid.org/000000021686778X",
        "https://orcid.org/0000000259219272",
        "https://orcid.org/0000000193420191",
        "https://orcid.org/0000000243595003",
        "https://orcid.org/0000000214435764",
        "https://orcid.org/0000000164480873",
        "https://orcid.org/0000000330729946",
        "https://orcid.org/0000000254529239",
        "https://orcid.org/0000000213235939",
        "https://orcid.org/0000000328525864",
        "https://orcid.org/000000030542119X",
        "https://orcid.org/0000000303787921",
        "https://orcid.org/0000000247988947",
        "https://orcid.org/0000000336590711",
        "https://orcid.org/0000000252162859",
        "https://orcid.org/0000000323459781",
        "https://orcid.org/0000000186947200",
        "https://orcid.org/000000017809771X",
        "https://orcid.org/0000000313224997",
        "https://orcid.org/0000000169068191",
        "https://orcid.org/0000000193961094",
        "https://orcid.org/0000000323711318",
        "https://orcid.org/0000000205601513",
        "https://orcid.org/0000000255707666",
        "https://orcid.org/0000000309002849",
        "https://orcid.org/0000000305923464",
        "https://orcid.org/0000000158792942",
        "https://orcid.org/0000000264259551",
        "https://orcid.org/0000000277954422",
        "https://orcid.org/0000000165586186",
        "https://orcid.org/0000000236925027",
        "https://orcid.org/0000000258625113",
        "https://orcid.org/0000000324936893",
        "https://orcid.org/000000020871664X",
        "https://orcid.org/0000000211734187",
        "https://orcid.org/0000000302522083"
      ],
      "creatorcontrib": [
        "Smith, Daniel G. A.",
        "Lolinco, Annabelle T.",
        "Glick, Zachary L.",
        "Lee, Jiyoung",
        "Alenaizan, Asem",
        "Barnes, Taylor A.",
        "Borca, Carlos H.",
        "Di Remigio, Roberto",
        "Dotson, David L.",
        "Ehlert, Sebastian",
        "Heide, Alexander G.",
        "Herbst, Michael F.",
        "Hermann, Jan",
        "Hicks, Colton B.",
        "Horton, Joshua T.",
        "Hurtado, Adrian G.",
        "Kraus, Peter",
        "Kruse, Holger",
        "Lee, Sebastian J. R.",
        "Misiewicz, Jonathon P.",
        "Naden, Levi N.",
        "Ramezanghorbani, Farhad",
        "Scheurer, Maximilian",
        "Schriber, Jeffrey B.",
        "Simmonett, Andrew C.",
        "Steinmetzer, Johannes",
        "Wagner, Jeffrey R.",
        "Ward, Logan",
        "Welborn, Matthew",
        "Altarawy, Doaa",
        "Anwar, Jamshed",
        "Chodera, John D.",
        "Dreuw, Andreas",
        "Kulik, Heather J.",
        "Liu, Fang",
        "Mart\u00ednez, Todd J.",
        "Matthews, Devin A.",
        "Schaefer, Henry F.",
        "\u0160poner, Ji\u0159\u00ed",
        "Turney, Justin M.",
        "Wang, Lee-Ping",
        "De Silva, Nuwan",
        "King, Rollin A.",
        "Stanton, John F.",
        "Gordon, Mark S.",
        "Windus, Theresa L.",
        "Sherrill, C. David",
        "Burns, Lori A."
      ],
      "issn": [
        "0021-9606",
        "1089-7690",
        "1520-9032",
        "1089-7690"
      ],
      "addtitle": [
        "J Chem Phys"
      ],
      "general": [
        "American Institute of Physics",
        "AIP Publishing LLC"
      ],
      "rsrctype": [
        "article"
      ],
      "startdate": [
        "20211128"
      ],
      "enddate": [
        "20211128"
      ],
      "scope": [
        "AAYXX",
        "CITATION",
        "NPM",
        "8FD",
        "H8D",
        "L7M",
        "7X8",
        "OTOTI",
        "5PM",
        "ACNBI",
        "ADTPV",
        "AOWAS",
        "D8T",
        "DF2",
        "ZZAVC",
        "ADTOC",
        "UNPAY"
      ],
      "description": [
        "Community efforts in the computational molecular sciences (CMS) are evolving toward modular, open, and interoperable interfaces that work with existing community codes to provide more functionality and composability than could be achieved with a single program. The Quantum Chemistry Common Driver and Databases (QCDB) project provides such capability through an application programming interface (API) that facilitates interoperability across multiple quantum chemistry software packages. In tandem with the Molecular Sciences Software Institute and their Quantum Chemistry Archive ecosystem, the unique functionalities of several CMS programs are integrated, including CFOUR, GAMESS, NWChem, OpenMM, Psi4, Qcore, TeraChem, and Turbomole, to provide common computational functions, i.e., energy, gradient, and Hessian computations as well as molecular properties such as atomic charges and vibrational frequency analysis. Both standard users and power users benefit from adopting these APIs as they lower the language barrier of input styles and enable a standard layout of variables and data. These designs allow end-to-end interoperable programming of complex computations and provide best practices options by default."
      ]
    },
    "delivery": {
      "fulltext": [
        "fulltext_multiple"
      ],
      "delcategory": [
        "Remote Search Resource"
      ]
    },
    "display": {
      "title": [
        "Quantum Chemistry Common Driver and Databases (QCDB) and Quantum Chemistry Engine (QCEngine): Automation and interoperability among computational chemistry programs"
      ],
      "type": [
        "article"
      ],
      "source": [
        "Unpaywall for CDI: Periodical Content",
        "AIP Journals (American Institute of Physics)"
      ],
      "creator": [
        "Smith, Daniel G. A. ; Lolinco, Annabelle T. ; Glick, Zachary L. ; Lee, Jiyoung ; Alenaizan, Asem ; Barnes, Taylor A. ; Borca, Carlos H. ; Di Remigio, Roberto ; Dotson, David L. ; Ehlert, Sebastian ; Heide, Alexander G. ; Herbst, Michael F. ; Hermann, Jan ; Hicks, Colton B. ; Horton, Joshua T. ; Hurtado, Adrian G. ; Kraus, Peter ; Kruse, Holger ; Lee, Sebastian J. R. ; Misiewicz, Jonathon P. ; Naden, Levi N. ; Ramezanghorbani, Farhad ; Scheurer, Maximilian ; Schriber, Jeffrey B. ; Simmonett, Andrew C. ; Steinmetzer, Johannes ; Wagner, Jeffrey R. ; Ward, Logan ; Welborn, Matthew ; Altarawy, Doaa ; Anwar, Jamshed ; Chodera, John D. ; Dreuw, Andreas ; Kulik, Heather J. ; Liu, Fang ; Mart\u00ednez, Todd J. ; Matthews, Devin A. ; Schaefer, Henry F. ; \u0160poner, Ji\u0159\u00ed ; Turney, Justin M. ; Wang, Lee-Ping ; De Silva, Nuwan ; King, Rollin A. ; Stanton, John F. ; Gordon, Mark S. ; Windus, Theresa L. ; Sherrill, C. David ; Burns, Lori A."
      ],
      "publisher": [
        "United States: American Institute of Physics"
      ],
      "ispartof": [
        "The Journal of chemical physics, 2021-11, Vol.155 (20), p.204801"
      ],
      "identifier": [
        "ISSN: 0021-9606",
        "ISSN: 1089-7690",
        "ISSN: 1520-9032",
        "EISSN: 1089-7690",
        "DOI: 10.1063/5.0059356",
        "PMID: 34852489",
        "CODEN: JCPSA6"
      ],
      "language": [
        "eng"
      ],
      "rights": [
        "Author(s)",
        "2021 Author(s). Published under an exclusive license by AIP Publishing.",
        "2021 Author(s). 2021 Author(s)"
      ],
      "snippet": [
        ".... The Quantum Chemistry Common Driver and Databases (QCDB) project provides such capability through an application programming interface (API..."
      ],
      "lds50": [
        "peer_reviewed"
      ],
      "oa": [
        "free_for_read"
      ],
      "description": [
        "Community efforts in the computational molecular sciences (CMS) are evolving toward modular, open, and interoperable interfaces that work with existing community codes to provide more functionality and composability than could be achieved with a single program. The Quantum Chemistry Common Driver and Databases (QCDB) project provides such capability through an application programming interface (API) that facilitates interoperability across multiple quantum chemistry software packages. In tandem with the Molecular Sciences Software Institute and their Quantum Chemistry Archive ecosystem, the unique functionalities of several CMS programs are integrated, including CFOUR, GAMESS, NWChem, OpenMM, Psi4, Qcore, TeraChem, and Turbomole, to provide common computational functions, i.e., energy, gradient, and Hessian computations as well as molecular properties such as atomic charges and vibrational frequency analysis. Both standard users and power users benefit from adopting these APIs as they lower the language barrier of input styles and enable a standard layout of variables and data. These designs allow end-to-end interoperable programming of complex computations and provide best practices options by default."
      ],
      "subject": [
        "Computational chemistry ;  Physics ;  Practice Guidelines as Topic ;  Quantum chemistry ;  Vibrational spectra"
      ],
      "keyword": [
        "Application programming interface ;  ARTICLES ;  Frequency analysis ;  Index Medicus ;  Interoperability ;  Software packages"
      ]
    },
    "addata": {
      "abstract": [
        "Community efforts in the computational molecular sciences (CMS) are evolving toward modular, open, and interoperable interfaces that work with existing community codes to provide more functionality and composability than could be achieved with a single program. The Quantum Chemistry Common Driver and Databases (QCDB) project provides such capability through an application programming interface (API) that facilitates interoperability across multiple quantum chemistry software packages. In tandem with the Molecular Sciences Software Institute and their Quantum Chemistry Archive ecosystem, the unique functionalities of several CMS programs are integrated, including CFOUR, GAMESS, NWChem, OpenMM, Psi4, Qcore, TeraChem, and Turbomole, to provide common computational functions, i.e., energy, gradient, and Hessian computations as well as molecular properties such as atomic charges and vibrational frequency analysis. Both standard users and power users benefit from adopting these APIs as they lower the language barrier of input styles and enable a standard layout of variables and data. These designs allow end-to-end interoperable programming of complex computations and provide best practices options by default."
      ],
      "jtitle": [
        "The Journal of chemical physics"
      ],
      "genre": [
        "article"
      ],
      "orcidid": [
        "https://orcid.org/0000-0002-7545-4416",
        "https://orcid.org/0000-0003-1322-4997",
        "https://orcid.org/0000-0003-2371-1318",
        "https://orcid.org/0000-0003-2852-5864",
        "https://orcid.org/0000-0001-7006-9378",
        "https://orcid.org/0000-0003-3072-9946",
        "https://orcid.org/0000-0002-1686-778X",
        "https://orcid.org/0000-0002-5452-9239",
        "https://orcid.org/0000-0003-0378-7921",
        "https://orcid.org/0000-0002-0871-664X",
        "https://orcid.org/0000-0001-8626-0900",
        "https://orcid.org/0000-0003-0592-3464",
        "https://orcid.org/0000-0003-2834-318X",
        "https://orcid.org/0000-0002-5570-7666",
        "https://orcid.org/0000-0002-1173-4187",
        "https://orcid.org/0000-0002-5921-9272",
        "https://orcid.org/0000-0002-3692-5027",
        "https://orcid.org/0000-0003-0542-119X",
        "https://orcid.org/0000-0003-0252-2083",
        "https://orcid.org/0000-0001-6906-8191",
        "https://orcid.org/0000-0002-1443-5764",
        "https://orcid.org/0000-0001-5879-2942",
        "https://orcid.org/0000-0002-9780-8123",
        "https://orcid.org/0000-0002-6425-9551",
        "https://orcid.org/0000-0001-7809-771X",
        "https://orcid.org/0000-0002-7795-4422",
        "https://orcid.org/0000-0001-6065-3167",
        "https://orcid.org/0000-0003-2345-9781",
        "https://orcid.org/0000-0003-2493-6893",
        "https://orcid.org/0000-0002-2460-8059",
        "https://orcid.org/0000-0003-2795-5483",
        "https://orcid.org/0000-0002-0560-1513",
        "https://orcid.org/0000-0003-0900-2849",
        "https://orcid.org/0000-0002-5862-5113",
        "https://orcid.org/0000-0002-2779-0749",
        "https://orcid.org/0000-0001-8694-7200",
        "https://orcid.org/0000-0003-3659-0711",
        "https://orcid.org/0000-0002-1323-5939",
        "https://orcid.org/0000-0003-1721-0330",
        "https://orcid.org/0000-0001-6558-6186",
        "https://orcid.org/0000-0002-4359-5003",
        "https://orcid.org/0000-0001-6448-0873",
        "https://orcid.org/0000-0001-9396-1094",
        "https://orcid.org/0000-0002-4798-8947",
        "https://orcid.org/0000-0001-9342-0191",
        "https://orcid.org/0000-0003-0683-7613",
        "https://orcid.org/0000-0002-5216-2859",
        "https://orcid.org/0000000224608059",
        "https://orcid.org/0000000297808123",
        "https://orcid.org/0000000275454416",
        "https://orcid.org/0000000327955483",
        "https://orcid.org/0000000306837613",
        "https://orcid.org/0000000317210330",
        "https://orcid.org/0000000186260900",
        "https://orcid.org/0000000160653167",
        "https://orcid.org/0000000227790749",
        "https://orcid.org/000000032834318X",
        "https://orcid.org/0000000170069378",
        "https://orcid.org/000000021686778X",
        "https://orcid.org/0000000259219272",
        "https://orcid.org/0000000193420191",
        "https://orcid.org/0000000243595003",
        "https://orcid.org/0000000214435764",
        "https://orcid.org/0000000164480873",
        "https://orcid.org/0000000330729946",
        "https://orcid.org/0000000254529239",
        "https://orcid.org/0000000213235939",
        "https://orcid.org/0000000328525864",
        "https://orcid.org/000000030542119X",
        "https://orcid.org/0000000303787921",
        "https://orcid.org/0000000247988947",
        "https://orcid.org/0000000336590711",
        "https://orcid.org/0000000252162859",
        "https://orcid.org/0000000323459781",
        "https://orcid.org/0000000186947200",
        "https://orcid.org/000000017809771X",
        "https://orcid.org/0000000313224997",
        "https://orcid.org/0000000169068191",
        "https://orcid.org/0000000193961094",
        "https://orcid.org/0000000323711318",
        "https://orcid.org/0000000205601513",
        "https://orcid.org/0000000255707666",
        "https://orcid.org/0000000309002849",
        "https://orcid.org/0000000305923464",
        "https://orcid.org/0000000158792942",
        "https://orcid.org/0000000264259551",
        "https://orcid.org/0000000277954422",
        "https://orcid.org/0000000165586186",
        "https://orcid.org/0000000236925027",
        "https://orcid.org/0000000258625113",
        "https://orcid.org/0000000324936893",
        "https://orcid.org/000000020871664X",
        "https://orcid.org/0000000211734187",
        "https://orcid.org/0000000302522083"
      ],
      "issn": [
        "0021-9606",
        "1089-7690",
        "1520-9032"
      ],
      "addtitle": [
        "J Chem Phys"
      ],
      "oa": [
        "free_for_read"
      ],
      "au": [
        "Smith, Daniel G. A.",
        "Lolinco, Annabelle T.",
        "Glick, Zachary L.",
        "Lee, Jiyoung",
        "Alenaizan, Asem",
        "Barnes, Taylor A.",
        "Borca, Carlos H.",
        "Di Remigio, Roberto",
        "Dotson, David L.",
        "Ehlert, Sebastian",
        "Heide, Alexander G.",
        "Herbst, Michael F.",
        "Hermann, Jan",
        "Hicks, Colton B.",
        "Horton, Joshua T.",
        "Hurtado, Adrian G.",
        "Kraus, Peter",
        "Kruse, Holger",
        "Lee, Sebastian J. R.",
        "Misiewicz, Jonathon P.",
        "Naden, Levi N.",
        "Ramezanghorbani, Farhad",
        "Scheurer, Maximilian",
        "Schriber, Jeffrey B.",
        "Simmonett, Andrew C.",
        "Steinmetzer, Johannes",
        "Wagner, Jeffrey R.",
        "Ward, Logan",
        "Welborn, Matthew",
        "Altarawy, Doaa",
        "Anwar, Jamshed",
        "Chodera, John D.",
        "Dreuw, Andreas",
        "Kulik, Heather J.",
        "Liu, Fang",
        "Mart\u00ednez, Todd J.",
        "Matthews, Devin A.",
        "Schaefer, Henry F.",
        "\u0160poner, Ji\u0159\u00ed",
        "Turney, Justin M.",
        "Wang, Lee-Ping",
        "De Silva, Nuwan",
        "King, Rollin A.",
        "Stanton, John F.",
        "Gordon, Mark S.",
        "Windus, Theresa L.",
        "Sherrill, C. David",
        "Burns, Lori A."
      ],
      "atitle": [
        "Quantum Chemistry Common Driver and Databases (QCDB) and Quantum Chemistry Engine (QCEngine): Automation and interoperability among computational chemistry programs"
      ],
      "date": [
        "2021-11-28"
      ],
      "risdate": [
        "2021"
      ],
      "volume": [
        "155"
      ],
      "issue": [
        "20"
      ],
      "spage": [
        "204801"
      ],
      "pages": [
        "204801-"
      ],
      "eissn": [
        "1089-7690"
      ],
      "coden": [
        "JCPSA6"
      ],
      "ristype": [
        "JOUR"
      ],
      "cop": [
        "United States"
      ],
      "pub": [
        "American Institute of Physics"
      ],
      "doi": [
        "10.1063/5.0059356"
      ],
      "pmid": [
        "34852489"
      ],
      "tpages": [
        "22"
      ],
      "format": [
        "journal"
      ]
    },
    "facets": {
      "creationdate": [
        "2021"
      ],
      "language": [
        "eng"
      ],
      "topic": [
        "Computational chemistry",
        "Physics",
        "Practice Guidelines as Topic",
        "Quantum chemistry",
        "Vibrational spectra"
      ],
      "prefilter": [
        "articles"
      ],
      "collection": [
        "CrossRef",
        "PubMed",
        "Technology Research Database",
        "Aerospace Database",
        "Advanced Technologies Database with Aerospace",
        "MEDLINE - Academic",
        "OSTI.GOV",
        "PubMed Central (Full Participant titles)",
        "SWEPUB Uppsala universitet full text",
        "SwePub",
        "SwePub Articles",
        "SWEPUB Freely available online",
        "SWEPUB Uppsala universitet",
        "SwePub Articles full text",
        "Unpaywall for CDI: Periodical Content",
        "Unpaywall"
      ],
      "toplevel": [
        "peer_reviewed",
        "online_resources"
      ],
      "frbrgroupid": [
        "cdi_FETCH-LOGICAL-b719t-84aa720b9d3572ebb76d4b776cd3edb3831f9ab035c8943fbdabd0222e226a5b3"
      ],
      "frbrtype": [
        "5"
      ],
      "jtitle": [
        "The Journal of chemical physics"
      ],
      "creatorcontrib": [
        "Smith, Daniel G. A.",
        "Lolinco, Annabelle T.",
        "Glick, Zachary L.",
        "Lee, Jiyoung",
        "Alenaizan, Asem",
        "Barnes, Taylor A.",
        "Borca, Carlos H.",
        "Di Remigio, Roberto",
        "Dotson, David L.",
        "Ehlert, Sebastian",
        "Heide, Alexander G.",
        "Herbst, Michael F.",
        "Hermann, Jan",
        "Hicks, Colton B.",
        "Horton, Joshua T.",
        "Hurtado, Adrian G.",
        "Kraus, Peter",
        "Kruse, Holger",
        "Lee, Sebastian J. R.",
        "Misiewicz, Jonathon P.",
        "Naden, Levi N.",
        "Ramezanghorbani, Farhad",
        "Scheurer, Maximilian",
        "Schriber, Jeffrey B.",
        "Simmonett, Andrew C.",
        "Steinmetzer, Johannes",
        "Wagner, Jeffrey R.",
        "Ward, Logan",
        "Welborn, Matthew",
        "Altarawy, Doaa",
        "Anwar, Jamshed",
        "Chodera, John D.",
        "Dreuw, Andreas",
        "Kulik, Heather J.",
        "Liu, Fang",
        "Mart\u00ednez, Todd J.",
        "Matthews, Devin A.",
        "Schaefer, Henry F.",
        "\u0160poner, Ji\u0159\u00ed",
        "Turney, Justin M.",
        "Wang, Lee-Ping",
        "De Silva, Nuwan",
        "King, Rollin A.",
        "Stanton, John F.",
        "Gordon, Mark S.",
        "Windus, Theresa L.",
        "Sherrill, C. David",
        "Burns, Lori A."
      ],
      "rsrctype": [
        "articles"
      ]
    },
    "sort": {
      "author": [
        "Smith, Daniel G. A. ; Lolinco, Annabelle T. ; Glick, Zachary L. ; Lee, Jiyoung ; Alenaizan, Asem ; Barnes, Taylor A. ; Borca, Carlos H. ; Di Remigio, Roberto ; Dotson, David L. ; Ehlert, Sebastian ; Heide, Alexander G. ; Herbst, Michael F. ; Hermann, Jan ; Hicks, Colton B. ; Horton, Joshua T. ; Hurtado, Adrian G. ; Kraus, Peter ; Kruse, Holger ; Lee, Sebastian J. R. ; Misiewicz, Jonathon P. ; Naden, Levi N. ; Ramezanghorbani, Farhad ; Scheurer, Maximilian ; Schriber, Jeffrey B. ; Simmonett, Andrew C. ; Steinmetzer, Johannes ; Wagner, Jeffrey R. ; Ward, Logan ; Welborn, Matthew ; Altarawy, Doaa ; Anwar, Jamshed ; Chodera, John D. ; Dreuw, Andreas ; Kulik, Heather J. ; Liu, Fang ; Mart\u00ednez, Todd J. ; Matthews, Devin A. ; Schaefer, Henry F. ; \u0160poner, Ji\u0159\u00ed ; Turney, Justin M. ; Wang, Lee-Ping ; De Silva, Nuwan ; King, Rollin A. ; Stanton, John F. ; Gordon, Mark S. ; Windus, Theresa L. ; Sherrill, C. David ; Burns, Lori A."
      ],
      "title": [
        "Quantum Chemistry Common Driver and Databases (QCDB) and Quantum Chemistry Engine (QCEngine): Automation and interoperability among computational chemistry programs"
      ],
      "creationdate": [
        "20211128"
      ]
    },
    "control": {
      "sourceid": [
        "proquest_unpay"
      ],
      "recordid": [
        "cdi_unpaywall_primary_10_1063_5_0059356"
      ],
      "sourcerecordid": [
        "2600467498"
      ],
      "originalsourceid": [
        "FETCH-LOGICAL-b719t-84aa720b9d3572ebb76d4b776cd3edb3831f9ab035c8943fbdabd0222e226a5b3"
      ],
      "recordtype": [
        "article"
      ],
      "addsrcrecordid": [
        "eNp9kstuEzEUhkcIRENhwQugEWwaRFpfZuwxC6SQlItUCVUCtpY94ySuZuzBl1R5Hx4UJ5MG2qKubNnf_5_j_zjLXkJwCgHBZ-UpACXDJXmUjSCo2IQSBh5nIwAQnDACyFH2zPsrAACkqHiaHeGiKlFRsVH2-zIKE2KXz1aq0z64TT6zXWdNPnd6rVwuTJPPRRBSeOXzk8vZ_ON4d3hfeG6W2qgtM-zG7_NpDLYTQSe_rUaboJztlRNStzpscpEqLfPadn0MO0y0eX0w7J1dOtH559mThWi9erFfj7Mfn86_z75MLr59_jqbXkwkhSxMqkIIioBkDS4pUlJS0hSSUlI3WDUSVxgumJAAl3XFCryQjZANQAgphIgoJT7O3g6-0fRicy3alvdOd8JtOAR8GzUv-T7qBL8bYH-t-igPpBWaz_XPKbduyWPkBUGYgIR_GPDEdqqplQlOtLdUt2-MXvGlXfOKwAIhlgxeDwbWB819rYOqV7U1RtWBwwojQGGCTvZVnP0VlQ88RVmrthVG2eg5IqAkANCd35s76JWNLg1gR4GC0IJViXr1b9uHfm9-UALOBqB21nunFjx1thtleoVu_xvc-I7ioZD3E_E3rgd4bd1fkPfN4iH4vvMfkQ8BYg"
      ],
      "sourcetype": [
        "Open Access Repository"
      ],
      "sourceformat": [
        "XML"
      ],
      "sourcesystem": [
        "Other"
      ],
      "pqid": [
        "2600467498"
      ],
      "iscdi": [
        "true"
      ],
      "pmcid": [
        "PMC8614229"
      ],
      "score": [
        "0.008109366"
      ]
    }
  },
  "delivery": {
    "link": [],
    "deliveryCategory": [
      "Remote Search Resource"
    ],
    "availability": [
      "fulltext_multiple"
    ],
    "displayLocation": false,
    "additionalLocations": false,
    "physicalItemTextCodes": "",
    "feDisplayOtherLocations": false,
    "displayedAvailability": "true",
    "holding": [],
    "almaOpenurl": "https://na05.alma.exlibrisgroup.com/view/uresolver/01SUNY_BUF/openurl?ctx_enc=info:ofi/enc:UTF-8&ctx_id=10_1&ctx_tim=2026-10-04 14:43:26&ctx_ver=Z39.88-2004&url_ctx_fmt=info:ofi/fmt:kev:mtx:ctx&url_ver=Z39.88-2004&rfr_id=info:sid/primo.exlibrisgroup.com-proquest_unpay&rft_val_fmt=info:ofi/fmt:kev:mtx:journal&rft.genre=article&rft.atitle=Quantum+Chemistry+Common+Driver+and+Databases+%28QCDB%29+and+Quantum+Chemistry+Engine+%28QCEngine%29%3A+Automation+and+interoperability+among+computational+chemistry+programs&rft.jtitle=The+Journal+of+chemical+physics&rft.au=Smith%2C+Daniel+G.+A.&rft.date=2021-11-28&rft.volume=155&rft.issue=20&rft.spage=204801&rft.pages=204801-&rft.issn=0021-9606&rft.eissn=1089-7690&rft.coden=JCPSA6&rft_id=info:doi/10.1063%2F5.0059356&rft.pub=American+Institute+of+Physics&rft.place=United+States&rft_id=info:pmid/34852489&rft_dat=<proquest_unpay>2600467498</proquest_unpay>&svc_dat=viewit&rft_pqid=2600467498"
  },
  "context": "PC",
  "adaptor": "Primo Central",
  "extras": {
    "citationTrails": {
      "citing": [
        "FETCH-LOGICAL-b719t-84aa720b9d3572ebb76d4b776cd3edb3831f9ab035c8943fbdabd0222e226a5b3"
      ],
      "citedby": [
        "FETCH-LOGICAL-b719t-84aa720b9d3572ebb76d4b776cd3edb3831f9ab035c8943fbdabd0222e226a5b3"
      ]
    },
    "timesCited": {}
  },
  "@id": "https://na05.alma.exlibrisgroup.com/primaws/rest/pub/pnxs/PC/cdi_unpaywall_primary_10_1063_5_0059356"
}
```

Create `tests/fixtures/onesearch_book_doc.json`:

```json
{
  "context": "L",
  "adaptor": "Local Search Engine",
  "@id": "https://na05.alma.exlibrisgroup.com/primaws/rest/pub/pnxs/L/990032130320204803",
  "pnx": {
    "display": {
      "source": [
        "Alma"
      ],
      "type": [
        "book"
      ],
      "language": [
        "eng"
      ],
      "title": [
        "Mental health : from policy to practice "
      ],
      "format": [
        "xiii, 330 pages : illustrations ; 24 cm"
      ],
      "identifier": [
        "$$CLC$$V  2008278257;$$CISBN$$V9780443103834;$$CISBN$$V0443103836;$$COCLC$$V(OCoLC)233263275;$$COCLC$$V(OCoLC)ocn233263275"
      ],
      "creationdate": [
        "2009"
      ],
      "lds08": [
        "BUF01003213032"
      ],
      "lds13": [
        "Sect. 1. Inclusion -- 1. Recovery and social inclusion: the changing mental health agenda / J. Repper and R. Perkins -- 2. User involvement in 21st century mental health services: 'This is our century' / A. Faulkner -- 3. Carers of people with mental health problems / J. Repper -- 4. Supporting equal citizenship and inclusion / E. Sayce -- Sect. 2. Specific service user groups -- 5. Black and minority ethnic issues / J. Bennett -- 6. Women and mental health / J. Carlisle -- 7. Implementing National Service Frameworks for mental health: dual diagnosis / H. Wells and R. Gray -- 8. Personality disorder in community settings / K. Dent-Brown, G. Parry and I. Kerr -- 9. Forensic service users / M. Number, J. Senior and J. Shaw -- Sect. 3. Service models -- 10. Assertive outreach / J. Freeman -- 11. Early interventions in psychosis / S. McGowan, Z. Iqbal and M. Birchwood -- 12. Acute in-patient care / A. Simpson -- 13. Community Mental Health Teams / J. Boardman -- 14. Mental health promotion / H. Guite and J. Bywaters -- 15. Primary care / D. Ekers and T. Ricketts -- 16. A national survey of Crisis Resolution Teams: an evaluation of implementation in England / S. Onyett and K. Linde -- Sect. 4. Supporting change -- 17. Research and development / M. Clark and C. Chilvers -- 18. NICE guidelines / K. Gournay -- 19. Developments in education and training: towards a whole system reform / D. Rushforth and I. Baguley -- 20. The National Service Framework for mental health: costs of implementation / M. Parsonage -- 21. The role of the voluntary sector / A. McCulloch and G. Howland -- 22. Mental health legislation / P. Bartlett -- Sect. 5. Conclusion -- 23. The National Service Framework for mental health: never mind the quality, feel the width? / C. Brooker and S. Onyett."
      ],
      "publisher": [
        "Edinburgh ; New York : Churchill Livingstone Elsevier"
      ],
      "description": [
        "\"In Mental Health: From Policy to Practice, the editors and contributors use their extensive experience and knowledge of mental health services to analyse the effectiveness of developments. They reflect on how policy has developed, what policy means for practice, to what extent policy has been implemented, and where the challenges lie.\"--BOOK JACKET."
      ],
      "mms": [
        "990032130320204803"
      ],
      "contributor": [
        "Brooker, Charlie (Professor of mental health and criminal justice)$$QBrooker, Charlie",
        "Repper, Julie.$$QRepper, Julie."
      ],
      "contents": [
        "Sect. 1. Inclusion -- 1. Recovery and social inclusion: the changing mental health agenda / J. Repper and R. Perkins -- 2. User involvement in 21st century mental health services: 'This is our century' / A. Faulkner -- 3. Carers of people with mental health problems / J. Repper -- 4. Supporting equal citizenship and inclusion / E. Sayce -- Sect. 2. Specific service user groups -- 5. Black and minority ethnic issues / J. Bennett -- 6. Women and mental health / J. Carlisle -- 7. Implementing National Service Frameworks for mental health: dual diagnosis / H. Wells and R. Gray -- 8. Personality disorder in community settings / K. Dent-Brown, G. Parry and I. Kerr -- 9. Forensic service users / M. Number, J. Senior and J. Shaw -- Sect. 3. Service models -- 10. Assertive outreach / J. Freeman -- 11. Early interventions in psychosis / S. McGowan, Z. Iqbal and M. Birchwood -- 12. Acute in-patient care / A. Simpson -- 13. Community Mental Health Teams / J. Boardman -- 14. Mental health promotion / H. Guite and J. Bywaters -- 15. Primary care / D. Ekers and T. Ricketts -- 16. A national survey of Crisis Resolution Teams: an evaluation of implementation in England / S. Onyett and K. Linde -- Sect. 4. Supporting change -- 17. Research and development / M. Clark and C. Chilvers -- 18. NICE guidelines / K. Gournay -- 19. Developments in education and training: towards a whole system reform / D. Rushforth and I. Baguley -- 20. The National Service Framework for mental health: costs of implementation / M. Parsonage -- 21. The role of the voluntary sector / A. McCulloch and G. Howland -- 22. Mental health legislation / P. Bartlett -- Sect. 5. Conclusion -- 23. The National Service Framework for mental health: never mind the quality, feel the width? / C. Brooker and S. Onyett."
      ],
      "place": [
        "Edinburgh ; New York :"
      ],
      "version": [
        "1"
      ],
      "lds03": [
        "Includes bibliographical references and index."
      ],
      "lds15": [
        "edited by Charlie Brooker, Julie Repper ; foreword by Martin Brown."
      ],
      "lds48": [
        "Brooker, Charlie.$$QBrooker, Charlie.",
        "Repper, Julie.$$QRepper, Julie."
      ],
      "lds100": [
        " / edited by Charlie Brooker, Julie Repper ; foreword by Martin Brown."
      ],
      "subject": [
        "Mental health services -- England -- Evaluation",
        "Mental health services -- Wales -- Evaluation",
        "Medical policy",
        "Mental Health Services -- organization & administration",
        "Program Evaluation",
        "Health Policy",
        "England",
        "Wales"
      ],
      "mesh": [
        "Mental Health Services -- organization & administration",
        "Program Evaluation",
        "Health Policy",
        "England",
        "Wales"
      ]
    },
    "control": {
      "sourcerecordid": [
        "990032130320204803"
      ],
      "recordid": [
        "alma990032130320204803"
      ],
      "sourceid": "alma",
      "originalsourceid": [
        "ocn233263275-01suny_network"
      ],
      "sourcesystem": [
        "WCM"
      ],
      "sourceformat": [
        "MARC21"
      ],
      "score": [
        12.16015925
      ],
      "isDedup": false
    },
    "addata": {
      "originatingSystemIDSubject": [
        "D004739",
        "fst01207649",
        "n80014574",
        "fst01219920",
        "D014852"
      ],
      "originatingSystemIDContributor": [
        "n87843207",
        "n96802239"
      ],
      "aulast": [
        "Brooker",
        "Repper"
      ],
      "aufirst": [
        "Charlie",
        "Julie."
      ],
      "auinit": [
        "C",
        "J"
      ],
      "addau": [
        "Brooker, Charlie",
        "Repper, Julie."
      ],
      "contributorfull": [
        "$$NBrooker, Charlie$$LBrooker$$FCharlie$$Rcontributor",
        "$$NRepper, Julie.$$Rcontributor"
      ],
      "date": [
        "2009"
      ],
      "isbn": [
        "9780443103834",
        "0443103836"
      ],
      "notes": [
        "Includes bibliographical references and index."
      ],
      "abstract": [
        "\"In Mental Health: From Policy to Practice, the editors and contributors use their extensive experience and knowledge of mental health services to analyse the effectiveness of developments. They reflect on how policy has developed, what policy means for practice, to what extent policy has been implemented, and where the challenges lie.\"--BOOK JACKET."
      ],
      "cop": [
        "Edinburgh"
      ],
      "pub": [
        "Churchill Livingstone Elsevier"
      ],
      "oclcid": [
        "(ocolc)233263275"
      ],
      "lccn": [
        "2008278257"
      ],
      "format": [
        "book"
      ],
      "genre": [
        "book"
      ],
      "ristype": [
        "BOOK"
      ],
      "btitle": [
        "Mental health : from policy to practice"
      ]
    },
    "sort": {
      "title": [
        "Mental health : from policy to practice /"
      ],
      "author": [
        "Brooker, Charlie (Professor of mental health and criminal justice)"
      ],
      "creationdate": [
        "2009"
      ]
    },
    "facets": {
      "frbrtype": [
        "6"
      ],
      "frbrgroupid": [
        "9011294523513946747"
      ]
    }
  },
  "delivery": {
    "bestlocation": {
      "isValidUser": true,
      "organization": "01SUNY_BUF",
      "libraryCode": "BUFLO",
      "availabilityStatus": "available",
      "subLocation": "Lockwood Library General Collection",
      "subLocationCode": "GEN",
      "mainLocation": "Lockwood Library",
      "callNumber": "RA790.7 .G7 M455 2009",
      "callNumberType": "0",
      "holdingURL": "OVP",
      "adaptorid": "ALMA_01",
      "ilsApiId": "990032130320204803",
      "holdId": "22246427660004803",
      "holKey": "HoldingResultKey [mid=22246427660004803, libraryId=158995240004803, locationCode=GEN, callNumber=RA790.7 .G7 M455 2009]",
      "matchForHoldings": [
        {
          "matchOn": "MainLocation",
          "holdingRecord": "852##b"
        }
      ],
      "stackMapUrl": "https://library.buffalo.edu/use/floor-plans/lockwood.html",
      "relatedTitle": null,
      "translateRelatedTitle": null,
      "yearFilter": null,
      "volumeFilter": null,
      "singleUnavailableItemProcessType": null,
      "boundWith": false,
      "@id": "_:0"
    },
    "holding": [
      {
        "isValidUser": true,
        "organization": "01SUNY_BUF",
        "libraryCode": "BUFLO",
        "availabilityStatus": "available",
        "subLocation": "Lockwood Library General Collection",
        "subLocationCode": "GEN",
        "mainLocation": "Lockwood Library",
        "callNumber": "RA790.7 .G7 M455 2009",
        "callNumberType": "0",
        "holdingURL": "OVP",
        "adaptorid": "ALMA_01",
        "ilsApiId": "990032130320204803",
        "holdId": "22246427660004803",
        "holKey": "HoldingResultKey [mid=22246427660004803, libraryId=158995240004803, locationCode=GEN, callNumber=RA790.7 .G7 M455 2009]",
        "matchForHoldings": [
          {
            "matchOn": "MainLocation",
            "holdingRecord": "852##b"
          }
        ],
        "stackMapUrl": "https://library.buffalo.edu/use/floor-plans/lockwood.html",
        "relatedTitle": null,
        "translateRelatedTitle": null,
        "yearFilter": null,
        "volumeFilter": null,
        "singleUnavailableItemProcessType": null,
        "boundWith": false,
        "@id": "_:0"
      }
    ],
    "electronicServices": null,
    "additionalElectronicServices": null,
    "filteredByGroupServices": null,
    "quickAccessService": null,
    "deliveryCategory": [
      "Alma-P"
    ],
    "serviceMode": [
      "ovp"
    ],
    "availability": [
      "available_in_library"
    ],
    "availabilityLinks": [
      "detailsgetit1"
    ],
    "availabilityLinksUrl": [],
    "displayedAvailability": null,
    "displayLocation": true,
    "additionalLocations": false,
    "titleRequestableAtItemLevel": false,
    "physicalItemTextCodes": null,
    "feDisplayOtherLocations": false,
    "almaInstitutionsList": [],
    "recordInstitutionCode": null,
    "recordOwner": "01SUNY_NETWORK",
    "hasFilteredServices": null,
    "digitalAuxiliaryMode": false,
    "digitalAuxiliaryThumbnail": false,
    "hideResourceSharing": false,
    "sharedDigitalCandidates": null,
    "consolidatedCoverage": null,
    "electronicContextObjectId": null,
    "almaOpenurl": null,
    "GetIt1": [
      {
        "category": "Alma-P",
        "links": [
          {
            "isLinktoOnline": false,
            "getItTabText": "service_getit",
            "adaptorid": "ALMA_01",
            "ilsApiId": "990032130320204803",
            "link": "OVP",
            "inst4opac": "01SUNY_BUF",
            "displayText": null,
            "@id": "_:0"
          }
        ]
      }
    ],
    "physicalServiceId": null,
    "link": [
      {
        "@id": ":_0",
        "linkType": "thumbnail",
        "linkURL": "https://proxy-na.hosted.exlibrisgroup.com/exl_rewrite/books.google.com/books?bibkeys=ISBN:0443103836,OCLC:,LCCN:2008278257&jscmd=viewapi&callback=updateGBSCover",
        "displayLabel": "thumbnail"
      }
    ],
    "hasD": null
  },
  "enrichment": {
    "virtualBrowseObject": {
      "isVirtualBrowseEnabled": true,
      "callNumber": "RA790.7 .G7 M455 2009",
      "callNumberBrowseField": "0"
    },
    "bibVirtualBrowseObject": {
      "isVirtualBrowseEnabled": true,
      "callNumber": "ra\"790.7 g7\u01c2m455 2009",
      "callNumberBrowseField": "LOC_CL"
    }
  }
}
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_parser.py` (needs `parse_brief_doc` already imported at the top of that
file, which it is):

```python
def test_parse_brief_doc_extracts_citation_fields_from_a_real_article():
    doc = _load("onesearch_article_doc.json")
    candidate = parse_brief_doc(doc)

    assert candidate.authors[0] == "Smith, Daniel G. A."
    assert len(candidate.authors) == 48
    assert candidate.journal_title == "The Journal of chemical physics"
    assert candidate.volume == "155"
    assert candidate.issue == "20"
    assert candidate.pages == "204801"  # single-page: trailing "-" stripped
    assert candidate.doi == "10.1063/5.0059356"
    assert candidate.publisher == "American Institute of Physics"
    assert candidate.isbn is None


def test_parse_brief_doc_extracts_citation_fields_from_a_real_book():
    doc = _load("onesearch_book_doc.json")
    candidate = parse_brief_doc(doc)

    assert candidate.authors == ["Brooker, Charlie", "Repper, Julie."]
    assert candidate.isbn == "9780443103834"
    assert candidate.publisher == "Churchill Livingstone Elsevier"
    assert candidate.place_of_publication == "Edinburgh"
    assert candidate.journal_title is None  # the book/article distinguisher
    assert candidate.doi is None


def test_parse_brief_doc_handles_page_range():
    doc = _load("onesearch_article_doc.json")
    doc["pnx"]["addata"]["spage"] = ["3276"]
    doc["pnx"]["addata"]["epage"] = ["3289"]
    candidate = parse_brief_doc(doc)
    assert candidate.pages == "3276-3289"


def test_parse_brief_doc_citation_fields_are_none_when_addata_absent():
    doc = {"pnx": {"display": {"title": ["X"]}, "control": {"recordid": ["r1"]}}}
    candidate = parse_brief_doc(doc)
    assert candidate.authors is None
    assert candidate.journal_title is None
    assert candidate.pages is None
```

- [ ] **Step 3: Run them to verify they fail**

```bash
.venv/bin/pytest tests/test_parser.py -v
```Expected: FAIL — `SearchCandidate() got an unexpected keyword argument 'authors'` (or an `AttributeError` if you write the parser calls before the model fields; either order fails first, that's fine).

- [ ] **Step 4: Add the new fields to `onesearch/models.py`**

Add these fields to the end of the `SearchCandidate` dataclass, right after
`availability_status: str | None` (defaults required: they must come after the existing
fields, which have no defaults):

```python
    # Citation data (agent/citation_tool.py), confirmed free in the brief response's
    # pnx.addata (no extra fetch): see onesearch/parser.py for real field provenance.
    authors: list[str] | None = None
    journal_title: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None
    isbn: str | None = None
    publisher: str | None = None
    place_of_publication: str | None = None
```

- [ ] **Step 5: Add the extraction helpers and wire them into `parse_brief_doc`**

In `onesearch/parser.py`, add these two helpers right before `def parse_brief_doc`:

```python
def _citation_authors(addata: dict) -> list[str] | None:
    # Confirmed live 2026-10-04: which key holds the author list varies by record,
    # not cleanly by type — a real journal article had "au" with "addau" absent;
    # a real book had "addau" with "au" absent. Both are already "Last, First"
    # strings (no $$Q markers), unlike display.creator.
    authors = addata.get("au") or addata.get("addau")
    return list(authors) if authors else None


def _citation_pages(addata: dict) -> str | None:
    spage = _first(addata.get("spage"))
    epage = _first(addata.get("epage"))
    if spage and epage:
        return f"{spage}-{epage}"
    if spage:
        return spage
    # Confirmed live 2026-10-04: a single-page article's "pages" value itself
    # carries a trailing "-" with no epage ("204801-"); strip it rather than
    # show a dangling dash in a citation.
    pages = _first(addata.get("pages"))
    return pages.rstrip("-") if pages else None
```

Then add these fields to the `SearchCandidate(...)` constructor call inside `parse_brief_doc`
(order doesn't matter, keyword arguments):

```python
        authors=_citation_authors(addata),
        journal_title=_first(addata.get("jtitle")),
        volume=_first(addata.get("volume")),
        issue=_first(addata.get("issue")),
        pages=_citation_pages(addata),
        doi=_first(addata.get("doi")),
        isbn=_first(addata.get("isbn")),
        publisher=_first(addata.get("pub")),
        place_of_publication=_first(addata.get("cop")),
```

- [ ] **Step 6: Run them to verify they pass**

```bash
.venv/bin/pytest tests/test_parser.py -v
```Expected: 13 passed (9 existing + 4 new).

- [ ] **Step 7: Run the full suite to confirm nothing else broke**

```bash
.venv/bin/pytest -q
```Expected: all pass (the new fields have defaults, so every existing `SearchCandidate(...)` call site, including every test fixture, keeps working unchanged).

- [ ] **Step 8: Commit**

```bash
git add onesearch/models.py onesearch/parser.py tests/test_parser.py tests/fixtures/onesearch_article_doc.json tests/fixtures/onesearch_book_doc.json
git commit -m "feat: parse citation fields (authors, journal, volume, DOI, ISBN, publisher) from real addata"
```

### Task 2: The formatter (`agent/citation_tool.py`)

Two real bugs were found and fixed while verifying this against live data before writing this
plan — both are encoded as regression tests below, not left to chance:
1. The Chicago article formatter interpolated `candidate.volume` directly into an f-string with
   no None-check; a real journal-title record (not a specific article — it has `journal_title`
   but no volume/issue/pages) rendered the literal word "None" in the citation.
2. The MLA article formatter always appended a comma after the year assuming page numbers would
   follow; when pages are missing, the citation dangled on a trailing comma with no closing
   period.

A third real, honest finding (not a bug — document it, don't "fix" it with a heuristic): some
OneSearch records' author field is an issuing organization, not a person ("International Center
of Mental Health Policy and Economics, issuing body"). The formatter has no way to distinguish
this from a personal "Last, First" name, so APA output can read oddly (an "i. b." initial from
"issuing body"). This is a known limitation of the simplification already documented in Global
Constraints, not something this task tries to detect and special-case.

**Files:**
- Create: `agent/citation_tool.py`
- Test: `tests/test_agent_citation_tool.py`

**Interfaces:**
- Consumes: `SearchCandidate` (Task 1).
- Produces: `CitationStyle = Literal["apa", "mla", "chicago"]`;
  `format_citation(candidate: SearchCandidate, style: CitationStyle) -> str` (never raises on
  missing fields — degrades to "Unknown author" / "n.d." rather than crashing).


- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent_citation_tool.py`:

```python
import json
from pathlib import Path

from agent.citation_tool import format_citation
from onesearch.models import SearchCandidate
from onesearch.parser import parse_brief_doc

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


def _candidate(**overrides):
    base = dict(
        record_id="1", title="A Title", author=None, resource_type=None, format=None,
        subject=None, description=None, date=None, source=None, delivery_category=None,
        online_link=None, call_number=None, library=None, sub_location=None,
        availability_status=None,
    )
    base.update(overrides)
    return SearchCandidate(**base)


ARTICLE = parse_brief_doc(_load("onesearch_article_doc.json"))
BOOK = parse_brief_doc(_load("onesearch_book_doc.json"))


def test_apa_article_has_authors_year_journal_volume_issue_pages_doi():
    citation = format_citation(ARTICLE, "apa")
    assert citation.startswith("Smith, D. G. A., et al. (2021).")
    assert "*The Journal of chemical physics*, *155*(20), 204801." in citation
    assert citation.endswith("https://doi.org/10.1063/5.0059356")


def test_mla_article_uses_quoted_title_and_vol_no_pp():
    citation = format_citation(ARTICLE, "mla")
    assert '"Quantum Chemistry Common Driver' in citation
    assert "vol. 155, no. 20, 2021, pp. 204801." in citation


def test_chicago_article_uses_volume_no_issue_parens_year_colon_pages():
    citation = format_citation(ARTICLE, "chicago")
    assert "*The Journal of chemical physics* 155, no. 20 (2021): 204801." in citation


def test_apa_book_has_two_authors_in_full_with_ampersand():
    citation = format_citation(BOOK, "apa")
    assert citation == "Brooker, C., & Repper, J. (2009). *Mental health : from policy to practice*. Churchill Livingstone Elsevier."


def test_mla_book_second_author_in_natural_order_with_marc_period_stripped():
    citation = format_citation(BOOK, "mla")
    assert "Brooker, Charlie, and Julie Repper." in citation
    assert "Julie. Repper" not in citation  # the MARC trailing-period artifact


def test_chicago_book_includes_place_of_publication():
    citation = format_citation(BOOK, "chicago")
    assert citation.startswith("Brooker, Charlie, and Julie Repper.")
    assert "Edinburgh: Churchill Livingstone Elsevier, 2009." in citation


def test_three_or_more_authors_collapse_to_et_al():
    for style in ("apa", "mla", "chicago"):
        citation = format_citation(ARTICLE, style)
        assert "et al." in citation
        assert "Lolinco" not in citation  # the 2nd of 48 real authors is not listed


def test_single_author_no_et_al():
    candidate = _candidate(title="Solo Work", authors=["Jones, Mary"], date="2020")
    for style in ("apa", "mla", "chicago"):
        assert "et al." not in format_citation(candidate, style)


def test_missing_authors_degrades_to_unknown_author_not_a_crash():
    candidate = _candidate(title="No Author Work", date="2020")
    for style in ("apa", "mla", "chicago"):
        citation = format_citation(candidate, style)
        assert "Unknown author" in citation
        assert "None" not in citation


def test_missing_date_gives_nd_with_no_double_period():
    candidate = _candidate(title="Undated Work", authors=["Jones, Mary"])
    for style in ("mla", "chicago"):
        citation = format_citation(candidate, style)
        assert citation.endswith("n.d.")
        assert "n.d.." not in citation


def test_missing_doi_omits_the_doi_url_entirely():
    article_no_doi = _candidate(
        title="An Article", authors=["Jones, Mary"], journal_title="Some Journal",
        volume="1", issue="2", pages="3-4", date="2020",
    )
    citation = format_citation(article_no_doi, "apa")
    assert "doi.org" not in citation
    assert citation.endswith("3-4.")


def test_book_fields_never_appear_in_an_article_citation_and_vice_versa():
    # ARTICLE has no isbn/publisher-as-book-field confusion: journal_title alone
    # must be what routes to the article branch, not the presence of a publisher
    # (both books and articles carry a "publisher"/addata.pub).
    assert "isbn" not in format_citation(ARTICLE, "chicago").lower()


def test_title_whitespace_from_source_data_is_stripped():
    assert "practice *" not in format_citation(BOOK, "apa")
    assert "practice*" in format_citation(BOOK, "apa")


def test_article_with_no_volume_issue_pages_never_leaks_a_literal_none():
    # Regression: confirmed live 2026-10-04 against a real journal-title record
    # (OneSearch can return a journal's own entry, not a specific article in it,
    # with journal_title set but no volume/issue/pages) — the Chicago formatter
    # used to interpolate a missing candidate.volume straight into an f-string,
    # rendering the literal word "None".
    candidate = _candidate(
        title="The journal of mental health policy and economics",
        journal_title="The journal of mental health policy and economics",
        authors=["Some Publisher, issuing body"],
    )
    for style in ("apa", "mla", "chicago"):
        citation = format_citation(candidate, style)
        assert "None" not in citation


def test_mla_article_with_no_pages_ends_cleanly_with_a_period_not_a_comma():
    # Regression: confirmed live 2026-10-04 — the year was always followed by a
    # comma assuming page numbers would come next; when pages are missing, the
    # citation dangled on a trailing comma with no closing period.
    candidate = _candidate(
        title="A Journal Entry", journal_title="Some Journal",
        authors=["Jones, Mary"], date="2020",
    )
    citation = format_citation(candidate, "mla")
    assert citation.endswith("2020.")
    assert not citation.endswith(",")
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/bin/pytest tests/test_agent_citation_tool.py -v
```Expected: FAIL, `ModuleNotFoundError: No module named 'agent.citation_tool'`.

- [ ] **Step 3: Write `agent/citation_tool.py`**

```python
from typing import Literal

from onesearch.models import SearchCandidate

CitationStyle = Literal["apa", "mla", "chicago"]

MAX_AUTHORS_IN_FULL = 2
"""Simplification, not full style-guide fidelity: real APA lists up to 20 authors
before "et al.", MLA/Chicago cut off differently. This project lists 1-2 authors
in full and uses "et al." from the 3rd author on, for every style."""


def _strip_marc_period(name: str) -> str:
    """Real source data can carry a trailing "." on a full first name as a MARC
    formatting artifact, not a true initial (confirmed live 2026-10-04: "Repper,
    Julie."). A real initial is one letter before the dot ("Julie K."); anything
    longer is the artifact, stripped so it doesn't look like a truncated name."""
    if name.endswith("."):
        word_before_dot = name[:-1].rsplit(" ", 1)[-1]
        if len(word_before_dot) > 1:
            return name[:-1]
    return name


def _split_name(full: str) -> tuple[str, str]:
    """"Last, First Middle" -> ("Last", "First Middle"). No comma -> (full, "")."""
    if "," in full:
        last, _, rest = full.partition(",")
        return last.strip(), _strip_marc_period(rest.strip())
    return full.strip(), ""


def _initials(first_middle: str) -> str:
    parts = first_middle.replace(".", " ").split()
    return " ".join(f"{p[0]}." for p in parts if p)


def _apa_name(full: str) -> str:
    last, first_middle = _split_name(full)
    initials = _initials(first_middle)
    return f"{last}, {initials}" if initials else last


def _natural_name(full: str) -> str:
    """"Last, First" -> "First Last", for a non-first MLA/Chicago author."""
    last, first_middle = _split_name(full)
    return f"{first_middle} {last}".strip() if first_middle else last


def _last_first_name(full: str) -> str:
    """"Last, First" kept as-is (trailing period artifacts from source data aside),
    for the first MLA/Chicago author."""
    last, first_middle = _split_name(full)
    return f"{last}, {first_middle}" if first_middle else last


def _authors_apa(authors: list[str]) -> str:
    if len(authors) == 1:
        return _apa_name(authors[0])
    if len(authors) == 2:
        return f"{_apa_name(authors[0])}, & {_apa_name(authors[1])}"
    return f"{_apa_name(authors[0])}, et al."


def _authors_mla_chicago(authors: list[str]) -> str:
    if len(authors) == 1:
        return _last_first_name(authors[0]) + "."
    if len(authors) == 2:
        return f"{_last_first_name(authors[0])}, and {_natural_name(authors[1])}."
    return f"{_last_first_name(authors[0])}, et al."


def _year(date: str | None) -> str:
    if date and len(date) >= 4 and date[:4].isdigit():
        return date[:4]
    return "n.d."


def _year_sentence(year: str) -> str:
    """`year` + a closing period, without doubling it when year is already
    "n.d." (which carries its own period)."""
    return year if year.endswith(".") else f"{year}."


def _join(*parts: str | None, sep: str = " ") -> str:
    """Joins only the pieces that are actually present, so a missing field never
    shows up as a literal "None" or a dangling separator."""
    return sep.join(p for p in parts if p)


def _is_article(candidate: SearchCandidate) -> bool:
    return bool(candidate.journal_title)


def _format_apa(candidate: SearchCandidate) -> str:
    authors = _authors_apa(candidate.authors) if candidate.authors else "Unknown author"
    year = _year(candidate.date)
    if _is_article(candidate):
        issue = f"({candidate.issue})" if candidate.issue else ""
        volume_issue = _join(f"*{candidate.volume}*" if candidate.volume else None, issue, sep="")
        locator = _join(volume_issue, candidate.pages, sep=", ")
        doi = f"https://doi.org/{candidate.doi}" if candidate.doi else None
        return _join(
            f"{authors} ({year}).",
            f"{candidate.title.strip()}.",
            _join(f"*{candidate.journal_title}*", locator, sep=", ") + ".",
            doi,
        )
    return _join(f"{authors} ({year}).", f"*{candidate.title.strip()}*.", f"{candidate.publisher}." if candidate.publisher else None)


def _format_mla(candidate: SearchCandidate) -> str:
    authors = _authors_mla_chicago(candidate.authors) if candidate.authors else "Unknown author."
    year = _year(candidate.date)
    if _is_article(candidate):
        vol = f"vol. {candidate.volume}," if candidate.volume else None
        no = f"no. {candidate.issue}," if candidate.issue else None
        pp = f"pp. {candidate.pages}." if candidate.pages else None
        year_piece = f"{year}," if pp else _year_sentence(year)
        return _join(authors, f'"{candidate.title.strip()}."', f"*{candidate.journal_title}*,", vol, no, year_piece, pp)
    return _join(authors, f"*{candidate.title.strip()}*.", f"{candidate.publisher}," if candidate.publisher else None, _year_sentence(year))


def _format_chicago(candidate: SearchCandidate) -> str:
    authors = _authors_mla_chicago(candidate.authors) if candidate.authors else "Unknown author."
    year = _year(candidate.date)
    if _is_article(candidate):
        no = f", no. {candidate.issue}" if candidate.issue else ""
        pp = f": {candidate.pages}" if candidate.pages else ""
        volume = f" {candidate.volume}" if candidate.volume else ""
        volume_part = f"*{candidate.journal_title}*{volume}{no} ({year}){pp}."
        return _join(authors, f'"{candidate.title.strip()}."', volume_part)
    publisher_part = _join(
        f"{candidate.place_of_publication}:" if candidate.place_of_publication else None,
        f"{candidate.publisher}," if candidate.publisher else None,
        _year_sentence(year),
    )
    return _join(authors, f"*{candidate.title.strip()}*.", publisher_part)


_FORMATTERS = {"apa": _format_apa, "mla": _format_mla, "chicago": _format_chicago}


def format_citation(candidate: SearchCandidate, style: CitationStyle) -> str:
    """A deterministic citation string in the given style. No LLM call. Missing
    fields are omitted cleanly rather than rendered as "None" or a stray comma."""
    return _FORMATTERS[style](candidate)
```

- [ ] **Step 4: Run them to verify they pass**

```bash
.venv/bin/pytest tests/test_agent_citation_tool.py -v
```Expected: 15 passed, including the 2 regression tests for the bugs described above.

- [ ] **Step 5: Spot-check the real rendered output once (manual, no assertions — just read it)**

```bash
.venv/bin/python - <<'EOF'
import json
from onesearch.parser import parse_brief_doc
from agent.citation_tool import format_citation

article = parse_brief_doc(json.load(open("tests/fixtures/onesearch_article_doc.json")))
book = parse_brief_doc(json.load(open("tests/fixtures/onesearch_book_doc.json")))
for label, candidate in [("ARTICLE", article), ("BOOK", book)]:
    for style in ("apa", "mla", "chicago"):
        print(f"{style.upper()} {label}:", format_citation(candidate, style))
        print()
EOF
```Expected: 6 citations, each reading as a real, usable citation with no literal "None",
no dangling comma, and titles/journal names in `*italic*` Markdown.

- [ ] **Step 6: Commit**

```bash
git add agent/citation_tool.py tests/test_agent_citation_tool.py
git commit -m "feat: add deterministic APA/MLA/Chicago citation formatter"
```

### Task 3: Attach citations to `compose()`'s output

**Files:**
- Modify: `agent/compose.py`, `agent/models.py`
- Test: `tests/test_agent_compose.py` (append)

**Interfaces:**
- Consumes: `format_citation` (Task 2).
- Produces: `build_citations(items: list[SelectionItem], candidates: list[SearchCandidate]) ->
  dict[str, dict[str, str]]` (`record_id -> {"apa": ..., "mla": ..., "chicago": ...}`, one entry
  per shown item — never looped over the full candidate list, same bounded-by-selection rule
  `fetch_detail()` already follows); `compose()`'s returned dict gains a `"citations"` key in
  every branch (`{}` when there are no candidates or confidence is low, the real built citations
  otherwise); `AgentState` gains `citations: dict[str, dict[str, str]]`.


- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_compose.py`:

```python
def test_compose_attaches_citations_for_every_shown_item():
    candidates = [
        _candidate("e1", title="Chemistry Basics", authors=["Doe, Jane"], date="2020"),
        _candidate("p1", title="Print Book", authors=["Smith, John"], date="2019"),
    ]
    model = FakeChatModel(Selection(
        items=[SelectionItem(record_id="e1", reason="r1"), SelectionItem(record_id="p1", reason="r2")],
        confidence="high",
    ))
    client = _client()

    out = compose(_state(candidates), model, client)

    assert set(out["citations"]) == {"e1", "p1"}
    assert set(out["citations"]["e1"]) == {"apa", "mla", "chicago"}
    assert "Doe" in out["citations"]["e1"]["apa"]


def test_compose_citations_empty_when_no_candidates():
    out = compose(_state([]), FakeChatModel(), _client())
    assert out["citations"] == {}


def test_compose_citations_empty_when_low_confidence():
    model = FakeChatModel(Selection(items=[], confidence="low"))
    out = compose(_state([_candidate("a")]), model, _client())
    assert out["citations"] == {}
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/bin/pytest tests/test_agent_compose.py -v
```Expected: FAIL — `KeyError: 'citations'` (the 3 new tests), existing tests unaffected.

- [ ] **Step 3: Add `citations` to `AgentState`**

In `agent/models.py`, add this line right after `selection: list[SelectionItem]`:

```python
citations: dict[str, dict[str, str]]  # record_id -> {style: citation text}
```

- [ ] **Step 4: Add `build_citations()` and wire it into `compose()`**

In `agent/compose.py`, add this import at the top:

```python
from agent.citation_tool import format_citation
```

Add this function right before `def compose(state: AgentState, model, client: OneSearchClient) -> dict:`:

```python
def build_citations(
    items: list[SelectionItem], candidates: list[SearchCandidate]
) -> dict[str, dict[str, str]]:
    """record_id -> {"apa": ..., "mla": ..., "chicago": ...}, one entry per shown
    item (never looped over the full candidate list — same bounded-by-selection
    rule fetch_detail() follows)."""
    by_id = {c.record_id: c for c in candidates}
    citations = {}
    for item in items:
        candidate = by_id.get(item.record_id)
        if candidate is None:
            continue
        citations[item.record_id] = {
            style: format_citation(candidate, style) for style in ("apa", "mla", "chicago")
        }
    return citations
```

Then in `compose()`, add `"citations": {}` to the two early-return dicts (the
"no candidates" branch and the "low confidence" branch), and change the final return line from:

```python
    answer = render_answer(items, candidates, details, plan.keywords, notes, client, rag_context)
    return {"selection": items, "details": details, "answer": answer}
```
to:

```python
    answer = render_answer(items, candidates, details, plan.keywords, notes, client, rag_context)
    citations = build_citations(items, candidates)
    return {"selection": items, "details": details, "answer": answer, "citations": citations}
```

- [ ] **Step 5: Run them to verify they pass**

```bash
.venv/bin/pytest tests/test_agent_compose.py -v
```Expected: 15 passed (12 existing + 3 new).

- [ ] **Step 6: Run the full suite**

```bash
.venv/bin/pytest -q
```Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add agent/compose.py agent/models.py tests/test_agent_compose.py
git commit -m "feat: attach APA/MLA/Chicago citations to every result compose() shows"
```

### Task 4: CLI rendering

**Files:**
- Modify: `agent/cli.py`
- Test: `tests/test_agent_cli.py`

**Interfaces:**
- Consumes: `state["citations"]` (Task 3).
- Produces: `run_chat(..., citations: bool = False)` (new parameter); `--citations` CLI flag,
  alongside the existing `--debug`.


- [ ] **Step 1: Update the existing `FakeGraph` fixture and write the failing tests**

In `tests/test_agent_cli.py`, add a `"citations"` key to `FakeGraph.invoke()`'s returned dict:

```python
"citations": {"r1": {"apa": "APA citation.", "mla": "MLA citation.", "chicago": "Chicago citation."}},
```

Then append these 2 tests to the end of the file:

```python
def test_citations_flag_prints_citations_for_each_result():
    read, write, written = _io("graduate", "q1", "/quit")

    run_chat(FakeGraph(), read=read, write=write, citations=True)

    assert any("[citations for r1]" in line for line in written)
    assert any("APA: APA citation." in line for line in written)


def test_citations_hidden_by_default():
    read, write, written = _io("graduate", "q1", "/quit")

    run_chat(FakeGraph(), read=read, write=write)

    assert not any("citations for" in line for line in written)
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/bin/pytest tests/test_agent_cli.py -v
```Expected: FAIL — `run_chat() got an unexpected keyword argument 'citations'`.

- [ ] **Step 3: Add the `citations` parameter and rendering to `run_chat()`**

Change the `run_chat` signature:

```python
def run_chat(graph, read=input, write=print, debug: bool = False, citations: bool = False) -> None:
```

Then, right after the existing `write(answer)` line inside the turn loop, add:

```python
        if citations:
            for record_id, styles in state.get("citations", {}).items():
                write(f"[citations for {record_id}]")
                for style, text in styles.items():
                    write(f"  {style.upper()}: {text}")
```

- [ ] **Step 4: Wire up the `--citations` flag in `main()`**

Add this argument right after the existing `--debug` one:

```python
parser.add_argument("--citations", action="store_true", help="show APA/MLA/Chicago citations for each result")
```

And change the `run_chat(...)` call at the end of `main()` to:

```python
run_chat(graph, debug=args.debug, citations=args.citations)
```

- [ ] **Step 5: Run them to verify they pass**

```bash
.venv/bin/pytest tests/test_agent_cli.py -v
```Expected: 9 passed.

- [ ] **Step 6: Run the full suite**

```bash
.venv/bin/pytest -q
```Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add agent/cli.py tests/test_agent_cli.py
git commit -m "feat: add --citations CLI flag"
```

### Task 5: Streamlit rendering + live verification + docs

**Files:**
- Modify: `streamlit_app/Home.py`, `CLAUDE.md`

**Interfaces:** none new; wires Task 3's data into the existing chat UI and records the real
findings from Tasks 2-5 in `CLAUDE.md`.

No `AppTest` suite exists for `Home.py` yet (that is part of the still-separate, not-yet-planned
"finish the Streamlit UI properly" work) — this step is verified live in a real browser, the same
pattern every prior Streamlit change in this project has used.


- [ ] **Step 1: Add the citation expanders**

In `streamlit_app/Home.py`, inside the `with st.chat_message("assistant", avatar="🦬"):` block
for a live turn, right after `st.write(answer)` and before the `if debug:` line, add:

```python
        titles_by_id = {c.record_id: c.title for c in state["search_result"].candidates}
        for record_id, styles in state.get("citations", {}).items():
            title = titles_by_id.get(record_id, record_id)
            with st.expander(f"Cite: {title}"):
                for style_name, text in styles.items():
                    st.markdown(f"**{style_name.upper()}**")
                    st.markdown(text)
```

- [ ] **Step 2: Run the full offline suite**

```bash
.venv/bin/pytest -q
```Expected: all pass (this step touches no tested logic, only Streamlit rendering).

- [ ] **Step 3: Verify live in a real browser (manual, needs `OPENAI_API_KEY` and network)**

```bash
.venv/bin/python -m pipeline.refresh   # only if vectorstore/ is stale
.venv/bin/streamlit run streamlit_app/Home.py --server.port 8502
```
Open http://localhost:8502, pick a role, click the "Peer-reviewed journals on mental health" chip.
Expected: once the answer renders, a "Cite: <title>" expander appears under it for each shown
result, collapsed by default. Open one: it shows APA, MLA, and Chicago, each with no literal
"None" and no dangling trailing comma (the two bugs Task 2 fixed) — confirmed live 2026-10-04 that
this exact query reliably hits that edge case (journal-title records with missing volume/issue),
so it is a real test of the fix, not a coincidence.

- [ ] **Step 4: Update `CLAUDE.md`**

Add a new subsection after the most recent Layer 2 slice section, documenting: the real addata
fields confirmed live (and that they're free — no extra fetch); the `au`/`addau` key-naming
surprise; the two real bugs found and fixed (literal "None", dangling MLA comma) with their
regression tests; the organizational-author known limitation; and why the conversational "cite
that" fast path from the original CLAUDE.md design was deferred in favor of inline rendering.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app/Home.py CLAUDE.md
git commit -m "feat: render citation expanders in the Streamlit chat UI"
```


---

## Slice acceptance (run after Task 5)

- [ ] `.venv/bin/pytest -q` passes with no network and no API key.
- [ ] A live Streamlit chat turn (the "Peer-reviewed journals on mental health" chip) shows a
      "Cite: <title>" expander per result, collapsed by default, with correct APA/MLA/Chicago
      text — no literal "None", no dangling comma.
- [ ] `python -m agent.cli --citations` shows the same 3 styles for a real turn's results.
- [ ] `git log --format=%B` on this slice's commits contains no `Co-Authored-By` or "Generated
      with" lines.
- [ ] Review, then `git push` once for the whole slice.
