from collections.abc import Callable
from copy import copy
from math import ceil
import re
import warnings

import pandas as pd
import pymupdf
from rapidfuzz import fuzz, utils, process


def parse_proudfoot(f: str) -> dict[str, dict[int, str]]:
    """
    Import the Proudfoot pdf and parse into Description and Title sections
    """
    doc = pymupdf.open(f)
    text = {"desc": {}, "titles": {}}

    for i, page in enumerate(doc): # type: ignore
        i = i + 1  # Just for ease when comparing indexing to the pdf pages
        # Printed page numbers are the numbers printed on the page (from title page, then i - 858)
        # Actual page numbers are the 1 - 886 numbers of the pages in the pdf, do not correspond to number on the page
        # Remember that all actual page numbers in the pdf are one greater than the Python indexing
        if i == 1:
            section = None
        if i == 126:  # Page 98
            section = "desc"
        if i == 596:  # Page 568
            section = None
        if i == 711:  # Page 683
            section = "titles"
        if i == 802:  # Page 774
            break

        if section:
            page_text = page.get_text() # get plain text (is in UTF-8)
            text[section][i] = page_text

    return text


def preprocess_text(text: dict[str, dict[int, str]]) -> dict[str, dict[int, str]]:
    """
    Correct errors in the OCR text so it's ready for use
    """
    # Description section

    # OCR error on page header
    if text["desc"][383][0] == "_":
        text["desc"][383] = text["desc"][383][10:]
    assert text["desc"][383][:5] == "DESCR"

    # Parse the first description page
    split = text["desc"][126].split("\n")
    if split[1] == "It should be assumed that the author/editor ":
        text["desc"][126] = "\n".join(split[9:49] + split[58:])
    assert text["desc"][126][:5] == "Abbas"

    # Titles section

    # bad reading order
    text["titles"][724] = text["titles"][724].replace("\nTITI..ES ", "")
    text["titles"][768] = text["titles"][768].replace("\nTfILES ", "")
    text["titles"][770] = text["titles"][770].replace("\nTI1LES ", "")
    text["titles"][774] = text["titles"][774].replace("\nTITLES ", "")

    # skip intro para from opening titles page
    if text["titles"][711][:20] == 'TITLES \nThe index at':
        text["titles"][711] = 'TITLES \n' + text["titles"][711][833:]

    # semi-accidental double column
    # the excluded section contains no main works among collected ceretera/cerita/ceritera/cetera
    if text["titles"][720][-5:] == "692 \n":
        text["titles"][720] = text["titles"][720][:1680]
        
    # accidental double column
    # the small amount of extracted is the only main work among collected ceretera/cerita/ceritera/cetera
    if text["titles"][721][:5] == '"Chre':
        text["titles"][721] = text["titles"][721][2613:2657].replace("\n", "")

    # accidental double columns
    # the small amount of extracted is the only main work among collected ceretera/cerita/ceritera/cetera
    if text["titles"][722][:5] == "TI1LE":
        text["titles"][722] = "Cerita Rampai-Rampai 1916 (t) - see also Abu Nawas 1917"

    return text


def preprocess_description_page(page: str, page_num: str|int) -> list[str]:
    """
    Apply preprocessing steps to description pages
    - trim whitespace
    - split on new lines
    - remove empty lines
    - remove header line if containing 'DESC' (Description) or 'EARL' (Early Malay Printed Books)
    - if page number appears exactly once then remove it
    """
    if type(page_num) == int:
        page_num = str(page_num)

    trim_space = page.replace(" \n", "\n")
    split = trim_space.split("\n")
    lines = [l for l in split if l]
    if "DESC" in lines[0] or "EARL" in lines[0]:
        lines = lines[1:]

    # remove if only present once
    if lines.count(page_num) == 1: # type: ignore
        lines.remove(page_num) # type: ignore

    return lines


def preprocess_titles_page(page: str, page_num: str|int) -> list[str]:
    """
    Apply preprocessing steps to description pages
    - if a date appears at the start of a line, append it to the previous line
    - remove new lines around hyphens/dashes
    - remove new lines before 'a' editions
    - split around remaining \n
    - remove blank lines
    - if page number appears exactly once then remove it
    - skip first line to remove EARLLY/TITLE header
    """
    if type(page_num) == int:
        page_num = page_num
    
    continuing_date_p = re.compile(r"\n(\d{4,4})")
    continue_date = continuing_date_p.sub(r"\1", page)  # \1 is the name for the matched date
    continue_dash = continue_date.replace("\n-\n", "- ").replace("-\n", "- ")
    continue_a = continue_dash.replace("\na ", "a ").replace("\na, ", "a, ")
    split = continue_a.split("\n")
    lines = [l.strip() for l in split if l]

    # For titles all page_num counts are 1
    if lines.count(page_num) == 1: # type: ignore
        lines.remove(page_num) # type: ignore
    
    # All first lines are EARLY or TITLES equivalents and can be dropped
    lines = lines[1:]
    return lines


def gen_title_lines(text: dict[str, dict[int, str]]) -> list[str]:
    """
    Convenience fn to preprocess and listify all titles from title pages
    """
    processed_title_pages = []
    for i in range(711, 802):
        page_num = str(i - 28)
        page = text["titles"][i]
        processed_title_pages.append(preprocess_titles_page(page=page, page_num=page_num))

    title_lines = []
    for p in processed_title_pages:
        title_lines.extend(p)
    
    return title_lines


def gen_desc_lines(text: dict[str, dict[int, str]]) -> tuple[list[str], dict[int, int]]:
    """
    Parse processed description pages into a list of all description lines and a mapping from lines to their page number
    
    :param text: Dictionary of EMP pages
    :type text: dict[str, dict[int, str]]
    :return: A tuple of all processed lines from the Description section and a lookup for those lines to the page they came from
    :rtype: tuple[list[str], dict[int, int]]
    """
    processed_desc_pages = []
    for i in range(126, 596):
        page_num = str(i - 28)
        page = text["desc"][i]
        processed_desc_pages.append(preprocess_description_page(page=page, page_num=page_num))

    description_lines = []
    line_page_lookup = {}
    line_count = 0
    for i, p in enumerate(processed_desc_pages):
        description_lines += p

        for j, _ in enumerate(p):
            line_page_lookup[j + line_count] = i + 98
        
        line_count += len(p)
    return description_lines, line_page_lookup


def manual_merge(all_titles: list[str], merge_file: str) -> list[str]:
    """
    Use manually created line merging list to merge titles that are incorrectly split across lines
    """
    with open(merge_file, encoding="utf8") as f:
        lines = [l.strip("\n").split("\t") for l in f.readlines()]
        bad_line_ids, bad_line_texts = [int(l[0]) for l in lines], [l[1] for l in lines]
    
    assert all([all_titles[line_id] == text for line_id, text in zip(bad_line_ids, bad_line_texts)])

    all_titles_concatenated = copy(all_titles)
    for l in bad_line_ids[::-1]:
        all_titles_concatenated[l-1] = all_titles_concatenated[l-1] + " " + all_titles_concatenated[l]

    all_titles_corrected = []
    for i, t in enumerate(all_titles_concatenated):
        if i not in bad_line_ids:
            all_titles_corrected += [t]
    
    return all_titles_corrected


def select_works(all_titles: list[str]) -> pd.DataFrame:
    """
    Select only works from the titles list that are represented in Description section
    Selection based on absence of 'see' or 'look' in title
    
    :param all_titles: List of all titles extracted from Titles section
    :type all_titles: list[str]
    :return: List of only main work titles represented in Descripion section
    :rtype: list[str]
    """
    see_re = re.compile(r"see(?! also)")

    works = []
    for title in all_titles:
        if see_re.search(title) or "look" in title:
            continue    
        else:
            works.append(title)

    # This cf is the only incorrect one not caught by the 'see' regex
    works.remove('Adab Kesopanan bagi Orang Muda-Muda Anak yang Bangsawan - cf Adab aI-Fatiy 1916')
    return pd.DataFrame(works, columns=["title"])


def shorten_proudfoot_title(title: str) -> str:
    """
    Shorten and clean a title from Proudfoot OCR
    - Remove trailing dates
    - Removing trailing 'a' editions

    :param title: Title to clean
    :type title: str
    :return: Cleaned title
    :rtype: str
    """
    works_date_re = re.compile(r"[ ±]{1,2}[l0-9]{4,4}")
    trailing_a_re = re.compile(r" a( |$)")

    no_date = re.split(works_date_re, title)[0]
    clean_short_title = re.split(trailing_a_re, no_date)[0]
    return clean_short_title


def shorten_aac_title(title: str) -> str:
    """
    Shorten and clean a title from AAC title list
    - Remove trailing dates
    - Removing trailing a|b|c editions
    
    :param title: AAC title to shorten and clean
    :type title: str
    :return: Cleaned title
    :rtype: str
    """
    works_date_re = re.compile(r"[ ±]{1,2}[l0-9]{4,4}")
    trailing_abc_re = re.compile(r" [abc] ?$")

    no_date = re.split(works_date_re, title)[0]
    no_abc_ed = re.split(trailing_abc_re, no_date)[0]
    clean_short_title = no_abc_ed
    return clean_short_title


def gen_short_titles(works: pd.DataFrame, converter: Callable[[str], str]) -> pd.DataFrame:
    """
    Convert a list of main works into their short titles equivalents
    
    :param main_works: List of main works
    :type main_works: list[str]
    :return: Short title equivalents for those works
    :rtype: Series[str]
    """
    if "title" not in works.columns:
        raise ValueError("`title` column missing")
    works["short_title_titles"] = works["title"].apply(converter)

    # Some work are duplicated due to line breaks converting "see <name of work>" to "see\n<name of work>", in which case the work gets picked up again
    if not works["short_title_titles"].is_unique:
        works = works.drop_duplicates(subset="short_title_titles").reset_index(drop=True)
    
    return works


def gen_aac_list(aac_file: str) -> pd.DataFrame:
    """
    Import the list of all AAC works
    
    :param aac_file: The filepath for the AAC list of works
    :type aac_file: str
    """
    aac_list = pd.read_csv(aac_file, header=None, names=["shelfmark", "short_title", "year"], usecols=[0,1,2])
    aac_list["short_title_no_year"] = aac_list["short_title"].apply(shorten_aac_title)
    return aac_list


def lookup_aac_titles(aac_df: pd.DataFrame, works: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Compare the list of AAC works to the list of main works to calculate how many works are represented
    
    :param aac_df: The AAC list of works
    :type aac_df: pd.DataFrame
    :param works: sorted list of Title works appearing in the AAC list
    :type works: list[tuple[str, str]]
    """
    matched_works = [(w, w) for w in aac_df["short_title_no_year"].unique() if w in works]
    missing_works = [w for w in aac_df["short_title_no_year"].unique() if w not in works]

    missing_work_matches = []
    for w in missing_works:
        matches = process.extract(w, works["short_title_titles"], scorer=fuzz.ratio, limit=3, processor=utils.default_process)
        missing_work_matches.append([w, matches])

    accepted_matches = []
    failed_matches = []
    for w, matches in missing_work_matches:
        if matches[0][1] >= 90:
            accepted_matches.append((w, matches[0][0], matches[0][1]))
        else:
            failed_matches.append((w, matches))

    matched_works += [(w[0], w[1]) for w in accepted_matches]
    return sorted(matched_works)


def find_nearest_line(row: pd.Series, desc_lines: list[str]) -> tuple[str, float, str]:
    """
    Function applied to each row of the works dataframe
    Use min_line/max_line to define a set of possible lines to contain a title
    Search for a short title within those lines
    
    :param row: Row from the dataframe of titles
    :type row: pd.Series
    :param desc_lines: Description
    :type desc_lines: list[str]
    :return: A tuple of the short title, the fuzz.ratio value for the match, and the matching title
    :rtype: tuple[str, float, str]
    """
    possible_lines = desc_lines[row["min_line"]:row["max_line"]]
    if row["entry_start"] is pd.NA:
        nearest_line = process.extract(row["short_title_titles"], possible_lines, scorer=fuzz.ratio, limit=1, processor=utils.default_process)[0]
        return (nearest_line[0], nearest_line[1], nearest_line[2] + row["min_line"])
    else:
        return (row["short_title_titles"], 100.0, row["entry_start"])


def gen_title_loc_df(works: pd.Series, desc_lines: list[str]) -> pd.DataFrame:
    """
    Create a dataframe of work titles and their matched aliases in the description section
    Matches only if the work string is present identically for a given entry
    
    :param works: Description
    :type works: pd.Series
    :param description_lines: Description
    :return: Description
    :rtype: DataFrame
    """
    title_loc = []
    title_line_tracker = 0  # This has to be accurate for it to work, otherwise can get too large too quickly

    for w in works:
        line_window = desc_lines[title_line_tracker: title_line_tracker + 2000]
        if w in line_window:
            line_loc = line_window.index(w) + title_line_tracker
            title_loc.append((w, None, line_loc, title_line_tracker, title_line_tracker + 2000))
            title_line_tracker = line_loc
        else:
            title_loc.append((w, None, None, title_line_tracker, title_line_tracker + 2000))

    title_loc_df = pd.DataFrame(title_loc, columns=["short_title_titles", "short_title_desc", "entry_start", "min_line", "max_line"])
    title_loc_df["entry_start"] = title_loc_df["entry_start"].astype("Int64")
    
    if not title_loc_df["entry_start"].dropna().is_monotonic_increasing:
        warnings.warn("Catalogue entry start lines do not monotonically increase. Check title dataframe sorting.", UserWarning)

    return title_loc_df


def apply_find_nearest(title_loc_df: pd.DataFrame, desc_lines: list[str]) -> pd.DataFrame:
    """
    Apply find_nearest across the titles df and split the outputs into named cols
    
    :param title_loc_df: Titles df
    :type title_loc_df: pd.DataFrame
    :param desc_lines: All Descriptions lines
    :type desc_lines: list[str]
    :return: Titles df with new columns for nearest matching titles in Descriptions section
    :rtype: DataFrame
    """
    nearest_apply = title_loc_df.apply(find_nearest_line, desc_lines=desc_lines, axis=1)
    title_loc_df["nearest_line"] = nearest_apply.apply(lambda x: x[0])
    title_loc_df["similarity"] = nearest_apply.apply(lambda x: x[1])
    title_loc_df["nearest_line_idx"] = nearest_apply.apply(lambda x: x[2])

    title_loc_df.loc[title_loc_df["similarity"] >= 90, "short_title_desc"] = title_loc_df.loc[title_loc_df["similarity"] >= 90, "short_title_titles"]
    title_loc_df.loc[title_loc_df["similarity"] >= 90, "entry_start"] = title_loc_df.loc[title_loc_df["similarity"] >= 90, "nearest_line_idx"]

    return title_loc_df


def gen_manual_check_df(title_loc_df: pd.DataFrame, line_page_lookup: dict[int, int]) -> pd.DataFrame:
    """
    Create the dataframe used to manually check uncertain matches between Titles and equivalents in the Description section

    :param title_loc_df: Titles df
    :type title_loc_df: pd.DataFrame
    :param line_page_lookup: Mapping between lines and their pages in the Description section
    :type line_page_lookup: dict[str, int]
    :return: Titles df with columns for finding minimum likely pages for titles in the Description section
    :rtype: DataFrame
    """
    missing_with_adjacent = []
    for t in title_loc_df.loc[title_loc_df["entry_start"].isna()].index:
        missing_with_adjacent += [t-1, t, t+1]

    blank_manual_check_df = title_loc_df.loc[sorted(list(set(missing_with_adjacent)))[:-1]]
    blank_manual_check_df["min_line_page"] = blank_manual_check_df["min_line"].map(line_page_lookup)
    
    return blank_manual_check_df


def extract_clean_entries(manual_check_df: pd.DataFrame, title_loc_df: pd.DataFrame, desc_lines: list[str]) -> pd.DataFrame:
    """
    Apply the info in a manual check csv for whether short titles mappings are correct to the titles df
    Check the integrity of the manual check csv
    Apply the manually checked lines to the titles df
    Apply a series of manual corrections for entry start/end points
    Extract entry text to new column
    
    :param manual_check_df: DataFrame created from a csv used to manually check whether fuzzy.ratio mappings for short titles are correct
    :type manual_check_df pd.DataFrame
    :param title_loc_df: The df of all titles and their locations in the Descriptions section
    :type title_loc_df: pd.DataFrame
    :param desc_lines: List of all Descriptions section lines 
    :type desc_lines: list[str]
    :rtype: DataFrame
    """
    # check all missing titles have been manually checked
    assert (~manual_check_df[manual_check_df["line_start"].isna()]["approve"].isna()).all()

    # check all line_start match nearest_line_idx
    assert (manual_check_df.dropna(subset="line_start")["line_start"].astype(int) == manual_check_df.dropna(subset="line_start")["nearest_line_idx"]).all()
    
    manually_approved_df = manual_check_df[manual_check_df["approve"] != -1]
    manually_approved_idx = manually_approved_df.index
    to_exclude_idx = manual_check_df[manual_check_df["approve"] == -1].index

    title_loc_df.loc[manually_approved_idx, "nearest_line"] = manually_approved_df["nearest_line_idx"].apply(lambda x: desc_lines[x])
    title_loc_df.loc[manually_approved_idx, "nearest_line_idx"] = manually_approved_df["nearest_line_idx"]

    title_loc_df.loc[manually_approved_idx, "entry_start"] = title_loc_df["nearest_line_idx"]
    title_loc_df.loc[manually_approved_idx, "short_title_desc"] = title_loc_df["nearest_line"]

    title_loc_df.drop(index=to_exclude_idx, inplace=True)
    
    title_loc_df["entry_end"] = title_loc_df["nearest_line_idx"].shift(-1).astype("Int64") - 1
    title_loc_df["correct_title"] = title_loc_df["short_title_titles"]
    
    title_loc_df.loc[title_loc_df.query("short_title_titles == 'I1mu Falak'").index, "correct_title"] = "Ilmu Falak"
    
    title_loc_df["short_title_desc"] = title_loc_df["short_title_desc"].str.strip('"')

    title_loc_df.loc[title_loc_df.query("short_title_desc == 'IlmuFalak'").index, "entry_start"] = 18278 - 2  # Fix an entry starting two lines late due to bad title OCR
    title_loc_df.loc[title_loc_df.query("short_title_desc == 'Ilmu Bintang'").index, "entry_end"] = 18277 - 2

    title_loc_df.loc[title_loc_df.query("short_title_desc == 'Sirat al-Mustakim'").index, "entry_start"] = 41676 - 1  # Fix an entry starting two lines late due to bad title OCR
    title_loc_df.loc[title_loc_df.query("short_title_desc == 'Slraj aI-KalbI'").index, "entry_end"] = 41675 - 1

    title_loc_df.loc[title_loc_df.query("short_title_desc == 'Akhbar'").index, "entry_end"] = 2520 - 64
    title_loc_df.loc[title_loc_df.query("short_title_desc == 'Akidat al-Munjian'").index, "entry_start"] = 2521 - 64  # Fix an entry starting late due to bad title OCR
    title_loc_df.loc[title_loc_df.query("short_title_desc == 'Akidat al-Munjian'").index, "entry_end"] = 2540 + 1  # Fix an entry starting late due to bad title OCR
    title_loc_df.loc[title_loc_df.query("short_title_desc == 'Alauddln'").index, "entry_start"] = 2541 + 1

    title_loc_df.loc[946, "entry_end"] = 51208  # Manually correct end of final entry

    title_loc_df["entry_text"] = title_loc_df.apply(lambda x: "\n".join(desc_lines[x["entry_start"]: x["entry_end"] + 1]), axis=1)

    return title_loc_df


def gen_prompt(entry_text: str, book_title: str) -> str:
    """
    Apply the info in a manual check csv for whether short titles mappings are correct to the titles df
    Check the integrity of the manual check csv
    Apply the manually checked lines to the titles df
    Apply a series of manual corrections for entry start/end points
    Extract entry text to new column
    
    :param entry_text: new line escaped text of an entry from the Descriptions section
    :type entry_text str
    :param book_title: The title of the work
    :type book_title: str
    :rtype: str
    """
    json_schema = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "object",
        "properties": {
            "editions": {
            "type": "array",
            "items": [
                {
                "type": "object",
                "properties": {
                    "edition_name": {
                    "type": "string"
                    },
                    "title": {
                    "type": "string"
                    },
                    "author": {
                    "type": "string"
                    },
                    "editor": {
                    "type": "string"
                    },
                    "translator": {
                    "type": "string"
                    },
                    "assistant_translator": {
                    "type": "string"
                    },
                    "proprietor": {
                    "type": "string"
                    },
                    "publisher": {
                    "type": "string"
                    },
                    "printer": {
                    "type": "string"
                    },
                    "copyist": {
                    "type": "string"
                    },
                    "contents": {
                    "type": "string"
                    },
                    "place_of_publication": {
                    "type": "string"
                    },
                    "printing_medium": {
                    "type": "string"
                    },
                    "script": {
                    "type": "string"
                    },
                    "dimensions": {
                    "type": "string"
                    },
                    "extent": {
                    "type": "string"
                    },
                    "Notes": {
                    "type": "string"
                    },
                    "References": {
                    "type": "string"
                    },
                    "Location": {
                    "type": "string"
                    },
                    "unclassified_text": {
                    "type": "string"
                    }
                },
                "required": [
                    "edition_name",
                    "title",
                    "author",
                    "editor",
                    "translator",
                    "assistant_translator",
                    "proprietor",
                    "publisher",
                    "printer",
                    "copyist",
                    "contents",
                    "place_of_publication",
                    "printing_medium",
                    "script",
                    "dimensions",
                    "extent",
                    "Notes",
                    "References",
                    "Location",
                    "unclassified_text"
                ]
                }
            ]
            }
        },
        "required": [
            "editions"
        ]
    }

    prompt = f"""Please extract structured metadata from the following text. The text is an entry for a particular book from a catalogue of books printed before 1925 in Malaysia.
    The text has been extracted from a pdf using optical character recognition and may contain errors. Do not correct these errors, but attempt to understand the correct words when extracting information.
    The text is split using line breaks. These separate lines in the OCR, but extra, unnecessary line breaks have sometimes been added between text from the same line.
    Each book entry begins with the book title, then is split into one or more editions. Each edition starts with an edition name in one of three formats:
    1) A year
    2) A year followed by a full stop then a letter (if there are multiple editions for one year)
    3) A letter (if the date of publication is unknown)

    The text for each edition normally reprints the edition date within it. The text for each edition contains different fields you should extract.
    These fields are marked by the field heading, and fields may run over multiple lines. All text before the next field heading belongs to that field. Not every entry has every field.
    Field headings are case insensitive. The Reference and Location fields are not usually followed by a colon. The other fields are followed by a colon.
    Sometimes fields are combined, such as 'author & proprietor', or 'publisher & printer'. In these cases repeat the information in text in the author and proprietor fields of the output.
    Field headings are:
    - author
    - editor
    - translator
    - assistant translator
    - proprietor
    - publisher
    - printer
    - copyist
    - contents
    - Notes
    - Reference(s)
    - Location(s)

    There is text between the edition name and the first field. There may also be text between fields that does not belong to that field. Both these types of text should be treated together as follows.
    This text may contain a title, a place of publication, the date of publication, the printing medium, the script of the text, the number of pages, the number of volumes, the dimensions of the edition.
    If the title is missing use the title provided later on in this prompt, otherwise use the title from the text. Use this text to extract the following fields:
    - title
    - place_of_publication
    - printing_medium
    - script
    - dimensions
    - extent (the number of volumes and number of pages) 
    
    The extent is number of pages, or if there are multiple volumes, then the number of volumes, sometimes called books, and the number of pages for each volume.
    If there are multiple volumes report the extent as '<XX> volumes; <YY> pages' where <XX> is the the total number of volumes for the work and <YY> is the total number of pages summed over all the volumes.

    The script of the text is the script the book itself is written in. If the word 'Jawi' is mentioned in the text return 'In Jawi script' for the script. Otherwise return <empty>.

    The printing_medium is how the text was printed. If the words 'lithographed' or 'lithographed jawi' are mentioned in the text return 'lithographed', otherwise return <empty>. 
    If the text contains 'lithographed illustrations' also return <empty>.

    Please extract the following information in json format. Only use the fields listed below. Not every entry has every field. If a field is missing represent it as <empty> in the output json.
    - edition name
    - title
    - author
    - editor
    - translator
    - assistant translator
    - publisher
    - printer
    - copyist
    - contents
    - place_of_publication
    - printing_medium
    - script
    - dimensions
    - extent (the number of volumes and number of pages) 
    - notes
    - references
    - locations
    Any text not in these fields include in the output json as a seperate field called 'unclassified_text'
        
    Please format the output as valid json using the schema below. Make sure to provide a valid and well-formatted JSON adhering to the given schema. Do not make up any information, only use what is provided in the text.
    {json_schema}    

    First, split the text into editions using the edition names, then assign the text for each edition to the appropriate fields.
    The title of this book is: {book_title}
    
    Book entry text:
    {entry_text}
    """
    # TODO refactor to include target edition in the output
    return prompt


def query_api(prompt, token):
    # probably done with a client
    # check the calls from the FF workshop
    # maybe async def
    pass


def log_api_call(prompt, output):
    pass


def extract_bl_shelfmark(locations_str: str) -> str:
    """
    Extract a BL shelfmark from a string containing the location of a work
    Return empty string if no BL shelfmarks present
    
    :param locations_str: Qwen extracted text containing holdings locations of a work
    :type locations_str: str
    """
    # These 4 patterns match all 686 
    three_part_re = re.compile(r"([o° 0-9lI]+[\.,])([a-z13]+[\.,])([ 0-9lI()]+)")
    jav_re = re.compile(r"Jav\. ?[\d()]+")
    orb_re = re.compile(r"ORB\. ?[0-9]+/[0-9]+")
    siam_re = re.compile(r"Siam \d+")

    locations = locations_str.split(";")
    bl_loc = [loc for loc in locations if "BL" in loc]
    if not bl_loc:
        return ""

    grp = three_part_re.search(bl_loc[0])
    if grp:
        p1, p2, p3 = grp.groups()
        if "l" in p1:
            p1 = p1.replace("l", "1")
        if "I" in p1:
            p1 = p1.replace("I", "1")
        
        if "1" in p2:
            p2 = p2.replace("1", "l")
        
        if "l" in p3:
            p3 = p3.replace("l", "1")
        if "I" in p3:
            p3 = p3.replace("I", "1")

        return p1.strip() + p2 + p3.strip()

    grp = jav_re.search(bl_loc[0])
    if grp:
        sm = grp.group()
        return sm
    
    grp = orb_re.search(bl_loc[0])
    if grp:
        sm = grp.group()
        return sm
    
    grp = siam_re.search(bl_loc[0])
    if grp:
        sm = grp.group()
        return sm

    return ""
        

def process_output_to_csv(json_dict: dict[str, dict[str, list[dict[str, str]]]]) -> pd.DataFrame:
    """
    Process Qwen JSON outputs to csv
    
    :param json_list: Description
    :type json_list: list[dict[str, str]]
    """
    metadata_lines = []
    for short_title, json in json_dict.items():
        for e in json["editions"]:
            e["edition_name"] = e["edition_name"].replace("t", "†")
            shelfmark = extract_bl_shelfmark(e["Location"])

            try:
                date = int(e["edition_name"].split(".")[0])
                date_1 = str(date)
                date_of_publication_in_arabic_or_roman_numerals = str(date)
                if date <= 1886:
                    method_of_acquisition = "purchased"
                elif date >= 1887:
                    method_of_acquisition = "legal deposit"
            except ValueError:
                date_1 = ""
                date_of_publication_in_arabic_or_roman_numerals = ""
                method_of_acquisition = ""

            edition = e["edition_name"]
            name = e["author"]
            extracted_title = e["title"]
            if extracted_title == "<empty>":
                extracted_title = short_title.replace("_", " ").title()
            place_of_publication = e["place_of_publication"]
            publisher = e["publisher"]
            extent = e["extent"]
            dimensions = e["dimensions"]
            general_notes = e["printing_medium"]
            bibliography_etc_note = f"Proudfoot 1993: {short_title} {e["edition_name"]}"
            unclassified_text = e["unclassified_text"]

            metadata = pd.DataFrame(
                data={
                    "short_title": short_title,
                    "edition": edition,
                    "shelfmark": shelfmark,
                    "date_1": date_1,
                    "name": name,
                    "title": extracted_title,
                    "place_of_publication": place_of_publication,
                    "publisher": publisher,
                    "date_of_publication_in_arabic_or_roman_numerals": date_of_publication_in_arabic_or_roman_numerals,
                    "extent": extent,
                    "dimensions": dimensions,
                    "general_notes": general_notes,
                    "bibliography_etc_note": bibliography_etc_note,
                    "method_of_acquisition": method_of_acquisition,
                    "unclassified_text": unclassified_text
                },
                index = [0]
            )

            metadata_lines.append(metadata)
    
    return pd.concat(metadata_lines).set_index(["short_title", "edition"], drop=True)


def map_orb_sm(series: pd.Series) -> pd.Series:

    map = {
        'ORB. 30/445 ': 'ORB. 30/445 (IOLR Malay F6 306/36.GF.7',
        'ORB. 30/446 ': 'ORB. 30/446 (IOLR Malay F6 306/36.G.8)',
        'ORB. 30/447 ': 'ORB. 30/447 (IOLR Malay F6 306/36.G.15)',
        'ORB. 30/448 ': 'ORB. 30/448 (IOLR Malay F6 306/36.G.16)',
        'ORB. 30/451 ': 'ORB. 30/451 (IOLR Malay B 306/36.F.29)',
        'ORB. 30/452 ': 'ORB. 30/452 (IOLR Malay B 306/36.F.40)',
        'ORB. 30/453 ': 'ORB. 30/453 (IOLR Malay 306/36.H.9)',
        'ORB. 30/457 ': 'ORB. 30/457 (IOLR Malay B 306/36.F. 16)',
        'ORB. 30/585': 'ORB. 30/585',
        'ORB. 30/611': 'ORB. 30/611',
        'ORB. 30/612': 'ORB. 30/612',
        'ORB.30/5553': 'ORB.30/5553',
        'ORB. 50/13': 'ORB. 50/13'
    }
    
    mapped_series = series.replace(map)

    return mapped_series

def post_process_extent(s: str) -> str:
    if "pp" in s[:10]:
        return s.split("pp")[0].replace("l", "1").replace("I", "1") + " pages"
    else:
        return s


def post_process_dimensions(s: str) -> str:
    height_str = s.split("x")[0]
    try:
        height = ceil(float(height_str))
        return str(height) + " cm"
    except ValueError:
        return height_str


def post_process_notes(s: str) -> str|int:
    if "lithographed" in s:
        return s
    else:
        return ""
    

def post_process_csv(metadata_df: pd.DataFrame, header_template: pd.DataFrame) -> pd.DataFrame:
    """
    Apply BL cataloguing standards in post-processing steps to metadata df
    
    :param metadata_df: Minimally modified Qwen output mapped onto target metadata fields
    :type metadata_df: pd.DataFrame
    :return: A dataframe aligned as closely as possible with BL/RDA cataloguing standards
    :rtype: DataFrame
    """
    header_template.set_index(pd.MultiIndex.from_arrays([(0,0), (0,1)], names=["short_title", "edition"]), inplace=True)

    marc_df = metadata_df.copy()
    marc_df.replace("<empty>", "", inplace=True)
    marc_df["shelfmark"] = map_orb_sm(marc_df["shelfmark"])
    marc_df["type_of_publication_date"] = "s"
    marc_df["country_of_publication"] = "si"
    marc_df["index"] = 0
    marc_df["main_language"] = "may"
    # TODO add Jawi check to Qwen prompt
    marc_df["type_of_name"] = "Personal name - surname first"
    marc_df["name"] = marc_df["name"].str.strip("[]").str.split(" :").apply(lambda x: x[0])
    marc_df["relationship_to_resource"] = "author"
    marc_df["title"] = marc_df["title"].str.replace('"', '')
    marc_df["title_ind2"] = 0
    marc_df["extent"] = marc_df["extent"].apply(lambda x: post_process_extent(x))
    marc_df["dimensions"] = marc_df["dimensions"].apply(lambda x: post_process_dimensions(x))
    marc_df["main_content"] = "text"
    marc_df["carrier_type"] = "volume"
    marc_df["general_notes"] = marc_df["general_notes"].apply(lambda x: post_process_notes(x))
    
    marc_df.rename(columns={
        'shelfmark': 'Shelfmark',
        'date_1': 'Date 1',
        'name': 'Name',
        'title': 'Title',
        'place_of_publication': 'Place of publication',
        'publisher': 'Publisher',
        'date_of_publication_in_arabic_or_roman_numerals': 'Date of publication in Arabic or Roman numerals',
        'extent': 'Extent',
        'dimensions': 'Dimensions',
        'general_notes': 'General notes',
        'bibliography_etc_note': 'Bibliography etc. note',
        'method_of_acquisition': 'Method of acquisition',
        'unclassified_text': 'unclassified_text',
        'type_of_publication_date': 'Type of publication date',
        'country_of_publication': 'Country of publication',
        'index': 'Index',
        'main_language': 'Main language',
        'type_of_name': 'Type of name',
        'relationship_to_resource': 'Relationship to resource',
        'title_ind2': ' ',
        'date_of_publication': 'Date of publication',
        'main_content': 'Main content type',
        'carrier_type': 'Carrier type'
    }, inplace=True)
    
    return pd.concat([header_template, marc_df])
