from collections.abc import Mapping
from functools import partial
import glob
import os
import re
from copy import copy
from xml.dom import minidom
import numpy as np
import pandas as pd
from pandas import Series
from langdetect import detect, LangDetectException
from tqdm import tqdm
from xml.etree import ElementTree as ET
tqdm.pandas()


def gen_xml_paths(path: str | os.PathLike) -> list[str]:
    """
    Collect xmls from a path
    Retries needed as this was originally on a network path that failed occasionally
    :param path: str : A location of transkribus model output xmls
    :return: list[str], list[str]
    """

    attempts = 0
    while attempts < 3:
        xmls = glob.glob(path) # type: ignore
        if xmls:
            break
        else:
            attempts += 1
            continue
    else:
        raise IOError(f"Failed to connect to {path}")

    return xmls


def gen_xml_trees(xmls: list[str]) -> dict[str, ET.ElementTree]:
    """
    Collect and correctly sort all the 2 col and 4 col xmls from a network root
    :param xmls: str : A list of locations of transkribus model output xmls
    :return: dict[xml.etree.ET]
    """
    xmls_sorted = sorted(xmls, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    xmlroots = {}

    for xml in tqdm(xmls_sorted):
        attempts = 0
        while attempts < 3:
            try:
                tree = ET.parse(xml)
                break
            except FileNotFoundError:
                attempts += 1
                continue
        else:
            raise FileNotFoundError(f"Failed to connect to: {xml}")
        root = tree.getroot()
        p = re.compile(r"BMC_\d{1,2}_[24]")
        n_cols = p.search(xml).group()[-1] # type: ignore
        xmlroots[os.path.basename(xml)[:-5] + f"_{n_cols}"] = root  # take the label that spans different sections of a volume

    return xmlroots


class TextLine(str):
    points = None


def extract_lines(root: ET.Element) -> list[TextLine]:
    """
    Extract the text lines from a page xml
    :param root: ET.Element: an xml root
    :return: list[str]: a list of the text lines in the xml
    """
    lines = []

    text_regions = [x for x in root[1] if len(x) > 2]  # Empty Text Regions Removed

    """"
    Tkb assigns reading order top to bottom then left to right
    RA/ID split Tkb pages into 2 or 4 'columns' - i.e. text regions
    2 column pages have 2 columns side by side, reading order is left col, right col 
    4 column pages have 2 columns at the top, a middle block of publisher text, then 2 cols at the bottom
    I.e. not 4 cols side by side, but 2 cols split horizontally and read top left, top right, bottom left, bottom right
    This code assumes initial reading order is TL, BL, TR, BR and changes to correct reading order 
    """
    if len(text_regions) % 2 == 0:
        half = int(len(text_regions) / 2)
        new_text_regions = []
        for x in range(half):
            new_text_regions.append(text_regions[x])
            new_text_regions.append(text_regions[x + half])
        text_regions = new_text_regions

    for text_region in text_regions:
        text_lines = text_region[1:-1]  # Skip coordinate data in first child
        for text_line in text_lines:
            points = [x[0].attrib['points'].split(" ")[::2] for x in text_line[2:-1]]  # only need 0th and 2nd
            points = [[(int(p.split(",")[0]), int(p.split(",")[1])) for p in line_points] for line_points in points]
            line = TextLine(text_line[-1][0].text)
            line.points = points # type: ignore
            lines.append(line)

    return [x for x in lines if x is not None]


def extract_lines_for_vol(vol: dict[str, ET.Element]) -> tuple[list[str], pd.DataFrame]:
    """
    Extract lines for a dict of xml roots
    :param vol: a dict of xml paths and corresponding ET roots
    :return:
    """
    lines = []
    xml_idx = []
    for xml, root in vol.items():
        root_lines = extract_lines(root)
        lines += root_lines
        xml_idx += [xml] * len(root_lines)
    xml_track_df = pd.DataFrame(
        data={
            "xml": xml_idx,
            "line": lines
        }
    )

    return lines, xml_track_df


# Regular expressions used to detect headings
caps_regex = re.compile("[A-Z][A-Z](?!I)[A-Z]+")

i_re = re.compile(r"""
 (?<![A-Za-z0-9\n\-\u201C.])   # Ensure no writing precedes re, effectively only allow a space
 (?<![(])I[ABC]                # Start chars for procter number of Grenville sm
 ([.,][ ]?[a-z0-9-/A]+)+       # 1+ repeats of [\.,] ?[a-z0-9-]+ i.e. the characters in the sm
 \**                           # allow trailing *
 ([. ]*[(][0-9-., ]+[)])?      # allow followed by a bracketed sets of numbers
 (?=[.,) ]|\Z)                 # lookahead for [\.,][ )] or end of string
""", re.VERBOSE)

g_re = re.compile(r"""
 (?<![A-Za-z0-9\n\-\u201C.])   # Ensure no writing precedes re, effectively only allow a space
 (?<=[( ])G                    # Start chars for procter number of Grenville sm
 ([.,][ ]?[a-z0-9-/A]+)+       # 1+ repeats of [\.,] ?[a-z0-9-]+ i.e. the characters in the sm
 \**                           # allow trailing *
 ([. ]*[(][0-9-., ]+[)])?      # allow followed by a bracketed sets of numbers
 (?=[.,) ]|\Z)                 # lookahead for [\.,][ )] or end of string
""", re.VERBOSE)

c_re = re.compile(r"""
 (?<![A-Za-z0-9\n\-\u201C.])   # Ensure no writing precedes re
 (?<=[( ])                     # Preceded by space or ( 
 C                             # C of a King Charles lib sm
 \.[ ]?[0-9]+                  # stop, optional space, 1+ number
 \.[ ]?[a-z]+                  # stop, optional space, 1+ letter
 ([.,][ ]?[0-9-*]+)+           # 1+ (stop/comma, optional space, 1+ number)  
 ([. ]*[(][0-9. ]+[)])?        # allow followed by a bracketed sets of numbers
 (?=[ )]|\Z)                   # lookahead for [ )] or end of string
""", re.VERBOSE)

one_num_regex = re.compile(r"1\.\s[a-z]")
date_regex = re.compile("1[45][0-9][0-9]")


def date_check(line: str) -> re.Match | str | bool:
    """
    Looks for identifying marks of a catalogue heading ending
    :param line:
    :return:
    """
    if line:
        return date_regex.search(line) or "Undated" in line
    else:
        return False


def _find_shelfmark(title: str, res: list[re.Pattern]) -> str | None:
    """
    Finds the associated title reference from a given line
    :param title:
    :param res: regexes - list[re.pattern]
    :return:
    """
    for re in res:
        if re.search(title):
            return re.search(title).group() # type: ignore


find_shelfmark = partial(_find_shelfmark, res=[i_re, g_re, c_re])


def find_headings(lines: list[str]) -> tuple[list[str], list[list[int]], list[str]]:
    """
    Finds all headings from a list of lines
    :param lines: list[str]
    :return: tuple[list[str], list[list[int]]
    """
    sm_titles = []  # The names of the titles
    title_indices = []
    ordered_lines = copy(lines)
    # TODO include the first catalogue entry as well
    for i, l in enumerate(lines):
        sm = find_shelfmark(l)
        if sm:
            title = [l]
            title_index = []
            j = 1
            while i + j < len(lines) and j < 8:
                title_part = lines[i + j]
                if find_shelfmark(title_part):  # If a new catalogue entry begins during the current title
                    break

                title.append(title_part)
                title_index.append(i + j)
                j += 1

                if date_check(title_part) and caps_regex.search(" ". join(title)):  # Date marks the end of a heading
                    sm_titles.append([sm, title])
                    if "Bought in" in title[1]:  # not .lower() - these "Bought in" should all be capitalised
                        sm, bought_in = lines[i], lines[i+1]
                        ordered_lines[i], ordered_lines[i+1] = bought_in, sm
                        title_indices.append(title_index[1:])
                    else:
                        title_indices.append(title_index)
                    break

    title_shelfmarks = [t[0] for t in sm_titles]

    return title_shelfmarks, title_indices, ordered_lines


def extract_catalogue_entries(lines: list[str],
                              title_indices: list[list[int]],
                              title_shelfmarks: list[str],
                              xml_track_df: pd.DataFrame) -> pd.DataFrame:
    """
    Use catalogue entry indices to extract from the main list of lines
    :param lines:
    :param title_indices:
    :param title_shelfmarks:
    :param xml_track_df:
    :return: pd.DataFrame
    """
    xmls = []
    xml_start_line = []
    shelfmarks = []
    entries = []

    for i, idx in enumerate(title_indices[:-1]):
        # take the idx[1] and title_indices[i+1] to exclude leading shelfmark and include trailing shelfmark
        # TODO fix this indexing for the first entry?
        entry = lines[idx[0]: title_indices[i + 1][0]]
        entries.append(entry)
        xml_info = xml_track_df.loc[idx[0]: title_indices[i + 1][0]].groupby(by="xml").count()
        if xml_info.shape[0] == 1:
            xmls.append([xml_info.index[0]])
            xml_start_line.append([len(entry)])
        else:
            xmls.append(xml_info.index.to_list())
            xml_start_line.append(xml_info["line"].cumsum().to_list())
        shelfmarks.append(title_shelfmarks[i + 1])

    # naively goes from last title to end of text to create last entry
    last_entry_start = title_indices[-1][0]
    last_entry = lines[last_entry_start:]
    entries.append(last_entry)
    xml_info = xml_track_df.loc[last_entry_start:].groupby(by="xml").count()
    if xml_info.shape[0] == 1:
        xmls.append([xml_info.index[0]])
        xml_start_line.append([len(last_entry)])
    else:
        xmls.append(xml_info.index.to_list())
        xml_start_line.append(xml_info["line"].cumsum().to_list())
    shelfmarks.append(find_shelfmark(" ".join(last_entry)))

    entry_df = pd.DataFrame(
        data={"xmls": xmls, "xml_start_line": xml_start_line, "shelfmark": shelfmarks, # "copy": 1,
              "entry": entries, "title": title_indices}
    )
    entry_df["entry_text"] = entry_df["entry"].apply(lambda x:"\n".join(x))
    entry_df.insert(loc=2, column="vol_entry_num", value=np.arange(len(entry_df)))
    entry_df["word_locations"] = entry_df["entry"].apply(lambda x: [y.points for y in x])

    return entry_df


def extract_another_copy():
    """

    :return:
    """
    vars = {
        # 'Another compartment',  This was part of the information rather than about another copy
        'Another copy',  # This one's ok, picks up "of leaves xx - yy"
        'Another copy.',  # This is the cannon one
        # 'Another edition'  This one was in the text and didn't refer to another entry
        # 'Another issue,'  This did refer to another copy, but was also in the main text, and that issue was listed separetly
    }
    return None


def detect_en(s: str) -> bool:
    """
    Error handled english language detection
    :param s: str
    :return: bool
    """
    try:
        return detect(s) == "en"
    except LangDetectException:
        return False


def _prepare_for_classification(entries: Series) -> Series:
    """
    Convert entries into a flattened series where each row is a single line
    Ready to be classified by detect_en
    :param entries: Series
    :return: Series
    """
    idx_base = entries.apply(len).reset_index()
    idx = idx_base.apply(lambda x: [x["index"]] * x["entry"], axis=1).sum()
    idx_tups = [(i, j) for i, j in zip(idx, [x for x in range(len(idx))])]
    multi_idx = pd.MultiIndex.from_tuples(idx_tups)

    lines = Series(data=entries.sum(), index=multi_idx)
    return lines


def _select_lang_in_group(group: Series, en=True) -> Series:
    """
    Designed to be used in groupby operation on a Series
    Always retains the first two lines of an entry (title lines)
    Even if no other lines in the entry are classified as English
    :param group: Series
    :return: Series
    """
    bool_rolling = group.rolling(window=2, closed='right').mean()
    bool_rolling.iloc[0] = 1  # set Title rows to en
    #  .rolling(window=2) uses the index of the second data point in the window as the index
    #  transitions to new language sections therefore treat the 1st line of that language as a boundary
    #  to handle this bffill with a limit of 1, then ffill the rest of the NAs
    en_sections = bool_rolling.where(bool_rolling != 0.5, pd.NA).bfill(limit=1).ffill().astype(bool)
    if en:
        return group[en_sections]
    else:
        return group[~en_sections]


def get_language_sections(entries: Series) -> tuple[Series, Series]:
    """
    Find the English parts of a Series of entries
    Use rolling to allow for single words classified as 'en' in a non-english block without breaking the block
    :param entries: Series
    :return: (Series, Series)
    """
    lines = _prepare_for_classification(entries)
    line_is_en = lines.progress_apply(detect_en)

    en_sections = line_is_en.groupby(level=0).apply(_select_lang_in_group, en=True)
    en_entries = lines.loc[en_sections.index.droplevel(level=0)].transform(lambda x: [x]).groupby(level=0).agg(lambda x: x.sum())

    non_en_sections = line_is_en.groupby(level=0).apply(_select_lang_in_group, en=False)
    raw_non_en_entries = lines.loc[non_en_sections.index.droplevel(level=0)].transform(lambda x: [x]).groupby(level=0).agg(lambda x: x.sum())
    non_en_entries = raw_non_en_entries.reindex(en_entries.index)
    non_en_entries[non_en_entries.isna()] = ""
    return en_entries, non_en_entries


# Returns the number of lines in a page which are too long
def num_outliers_for_page(lines, std, mean, threshold=2):
    lengths = [len(x.split()) for x in lines if x is not None]
    lengths = [(x - mean) for x in lengths]
    lengths = [(x / std) for x in lengths]
    num_outliers = len([x for x in lengths if x > threshold])
    return num_outliers


# Find all of the poorly scanned pages in the input
def get_poorly_scanned_pages(volume_root, file_names):
    poorly_scanned_page_nums = []

    # Get all the lines for the volume and find the mean and std for the line lengths across all volumes
    vol_lines, xml_check_df = extract_lines_for_vol(volume_root)
    lengths = [len(x.split()) for x in vol_lines if x is not None]
    mean = np.mean(lengths)
    std = np.std(lengths)

    # For every page (xmlroot) in the volume
    for root, filename in zip(volume_root.values(), file_names):
        page = root
        page_lines = extract_lines(page)
        num_outliers = num_outliers_for_page(page_lines, std, mean)
        if num_outliers > 5:
            poorly_scanned_page_nums.append(filename.decode("utf-8"))
    return poorly_scanned_page_nums


# Save poorly scanned page numbers to a text file
def save_poorly_scanned_pages(poorly_scanned, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    save_path_file = os.path.join(out_path, "poorlyscanned.txt")
    with open(save_path_file, "w", encoding="utf-8") as f:
        for scan in poorly_scanned:
            f.write(scan + "\n")


def generate_xml(lines: list[str],  # TODO update to work with df output
                 title_indices: list[list[int]],
                 title_refs: list[str]) -> minidom.Document:
    """
    Generates an XML document based on the found catalogue headings in the document
    :param lines:
    :param title_indices:
    :param title_refs:
    :return:
    """
    xml = minidom.Document()
    text = xml.createElement('text')

    # TODO replace iteration through title indices with entries in rows of df
    for i, idx in tqdm(enumerate(title_indices[:-1]), total=len(title_indices) - 1):
        catalogue_indices = [x for x in range(idx[1], title_indices[i + 1][0])]
        full_title = "".join([lines[x] for x in idx])

        catalogue_entry = xml.createElement('catalogue_entry')
        catalogue_entry.setAttribute("SHELFMARK", title_refs[i + 1])
        catalogue_entry.setAttribute("HEADING", full_title)

        for catalogue_index in catalogue_indices:
            line = xml.createElement('line')
            line.setAttribute("CONTENT", lines[catalogue_index])
            catalogue_entry.appendChild(line)

        text.appendChild(catalogue_entry)

    xml.appendChild(text)
    return xml


# Saves the generated XML for the headings into a chosen location
def save_xml(lines: list[str],
             title_indices: list[list[int]],
             title_refs: list[str],
             out_path: str) -> None:

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    xml = generate_xml(lines, title_indices, title_refs)
    xml_str = xml.toprettyxml(indent="\t")
    save_path_file = out_path + "/headings.xml"
    with open(save_path_file, "w", encoding="utf-8") as f:
        f.write(xml_str)
        f.close()

    return None
