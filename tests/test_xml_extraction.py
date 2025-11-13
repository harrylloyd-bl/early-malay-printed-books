import emp.xml_extraction as xmle
from xml.etree import ElementTree as ET
import os
import glob
import pytest
from pandas import Series
from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) # type: ignore


@pytest.fixture()
def xml_roots():
    xml_roots = {}
    for xml in glob.glob("tests\\small_*.xml"):
        tree = ET.parse(xml)
        root = tree.getroot()
        xml_roots[os.path.basename(xml)[:-4]] = root
    return {k: xml_roots[k] for k in sorted(xml_roots)}


def test_extract_lines(xml_roots):
    lines = xmle.extract_lines(xml_roots["small_xml_example_1"])

    assert len(lines) == 3
    assert lines[0] == "Test line 1"


def test_extract_lines_text_region_order():
    four_region_xml = ET.parse("tests\\4_TextRegion_xml_example.xml")
    four_region_root = four_region_xml.getroot()
    ordered_lines = xmle.extract_lines(four_region_root)

    assert ordered_lines[0] == "Test line 1"
    assert ordered_lines[3] == "Test line 4"


def test_extract_lines_for_vol(xml_roots):
    all_lines, xml_track_df = xmle.extract_lines_for_vol(xml_roots)

    assert len(all_lines) == 6
    assert all_lines[-1] == "Test line 6"

    assert xml_track_df.shape == (6, 2)  # must be same length as all_lines
    assert xml_track_df['xml'].tolist() == [
        "small_xml_example_1", "small_xml_example_1", "small_xml_example_1",
        "small_xml_example_2", "small_xml_example_2", "small_xml_example_2"
    ]


def test_find_shelfmark():
    ia_sm = xmle.find_shelfmark("IA. 33.")
    ic_sm = xmle.find_shelfmark("IC. 3. a. 4.")
    i_letter_sm = xmle.find_shelfmark("IC. a3.")
    preceding_letter_i_sm = xmle.find_shelfmark("ASDFIB. 1345.")

    c_brackets_sm = xmle.find_shelfmark("(C.44.a.1).")
    c_space_sm = xmle.find_shelfmark(" C. 15632.a.3 ")
    preceding_letter_c_sm = xmle.find_shelfmark("aC. 15632.")

    bad_lower = xmle.find_shelfmark("asdf")
    bad_caps = xmle.find_shelfmark("NOTHINGTOSEEHERE")

    assert ia_sm == "IA. 33"
    assert ic_sm == "IC. 3. a. 4"
    assert i_letter_sm == "IC. a3"
    assert preceding_letter_i_sm is None

    assert c_brackets_sm == "C.44.a.1"
    assert c_space_sm == "C. 15632.a.3"
    assert preceding_letter_c_sm is None

    assert not bad_lower
    assert not bad_caps


def test_date_check():
    date_check = xmle.date_check("1409")
    untitled_check = xmle.date_check("Undated")
    bad_check = xmle.date_check("0300")

    assert date_check
    assert untitled_check
    assert not bad_check


# separate so easier to see their values in tests
fix_in_xml = ET.parse("tests\\title_xml_example_1.xml")
fix_root = fix_in_xml.getroot()
fix_lines = xmle.extract_lines(fix_root)
fix_titles, fix_title_indices = ["IA. 123.TITLE1456", "IA. 789.SECONDTITLE1506"], [[0, 1, 2], [6, 7, 8]]


@pytest.fixture()
def lines():
    return fix_lines

@pytest.fixture()
def titles():
    return fix_titles

@pytest.fixture()
def indices():
    return fix_title_indices


def test_find_headings(xml_roots, lines, titles, indices):
    t, i, l = xmle.find_headings(["asdf"])
    assert not t
    assert not i
    assert l == ["asdf"]

    no_title_lines, xml_track_df = xmle.extract_lines_for_vol(xml_roots)
    no_titles, no_all_title_indices, no_ordered_lines = xmle.find_headings(no_title_lines)
    # should both be empty
    assert not no_titles
    assert not no_all_title_indices
    assert len(no_ordered_lines) == 6

    """"
    the xml used to make title_indices contains three IA. numbers that could start titles
    The second IA. (IA. 456) has its title interrupted by IA. 789 before a date is found
    This tests the 
    `if shelfmark_check(title_part):
    `    break
    section of find_headings 
    """

    # lines is produced from title_xml_example_1.xml rather than the small_xml_examples
    title_shelfmarks, indices, o_l = xmle.find_headings(lines)

    assert len(title_shelfmarks), len(indices) == (2, 2)
    assert title_shelfmarks == ["IA. 123", "IA. 789"]
    assert indices == [[2, 3], [8, 9]]
    assert len(o_l) == 13
    assert (o_l[0], o_l[1], o_l[-1]) == ("Bought in 1456", "IA. 123.", "some line text")


def test_extract_catalogue_entries():
    root_1 = ET.parse("tests\\title_xml_example_1.xml").getroot()
    lines, xml_track_df = xmle.extract_lines_for_vol({"title_xml_example_1": root_1})
    title_shelfmarks, title_indices, o_l = xmle.find_headings(lines)
    catalogue_entries = xmle.extract_catalogue_entries(o_l, title_indices, title_shelfmarks, xml_track_df)

    assert catalogue_entries.shape == (2, 8)
    assert catalogue_entries.columns.tolist() == ["xmls", "xml_start_line", "vol_entry_num", "shelfmark", "entry", "title", "entry_text", "word_locations"]  # "copy"
    assert catalogue_entries["xmls"].tolist() == [["title_xml_example_1"], ["title_xml_example_1"]]
    assert catalogue_entries["xml_start_line"].to_list() == [[6], [5]]
    assert catalogue_entries["shelfmark"].tolist() == ["IA. 789", "IA. 353"]
    # assert catalogue_entries["copy"].sum() == catalogue_entries.shape[0]
    assert catalogue_entries["entry"].transform(len).tolist() == [6, 5]
    assert catalogue_entries.loc[0, "entry"][0] == "TITLE" # type: ignore
    assert catalogue_entries.loc[0, "entry"][-1] == "IA. 789." # type: ignore
    assert catalogue_entries.loc[1, "entry"][0] == "SECONDTITLE" # type: ignore
    assert catalogue_entries.loc[1, "entry"][-1] == "some line text" # type: ignore

    # entries across xmls
    root_2 = ET.parse("tests\\title_xml_example_2.xml").getroot()
    lines, xml_track_df = xmle.extract_lines_for_vol({"title_xml_example_1": root_1, "title_xml_example_2": root_2})
    title_shelfmarks, title_indices, o_l = xmle.find_headings(lines)
    catalogue_entries = xmle.extract_catalogue_entries(o_l, title_indices, title_shelfmarks, xml_track_df)

    assert catalogue_entries.shape == (3, 8)
    assert catalogue_entries["xmls"].tolist() == [["title_xml_example_1"], ["title_xml_example_1"], ["title_xml_example_1", "title_xml_example_2"]]
    assert catalogue_entries["xml_start_line"].to_list() == [[6], [3], [2, 4]]


def test_detect_en():
    assert xmle.detect_en("Hello this should be English")
    assert not xmle.detect_en("Bonjour je m'appelle Jean-Paul")
    assert not xmle.detect_en(".")


def test_get_english_sections():
    entries = Series(
        data=[  # each list represents a catalogue entry, each list item a line in that entry line
            ["This statement is English", "Ce ligne est Francais", "and English again"],
            [
                "Ce ligne est Francais", "Ce ligne est Francais une", "Ce ligne est Francais deux",
                "This statement is English", "Ce ligne est Francais", "This statement is English one",
                "This statement is English", "1.2.4", "This statement is English"]
        ],
        index=[0, 1], name="entry"
    )

    expected_en_result = Series(
        data=[
            ["This statement is English", "Ce ligne est Francais", "and English again"],
            [
                "Ce ligne est Francais", "This statement is English one",
                "This statement is English", "1.2.4", "This statement is English"
            ]
        ],
        index=[0, 1], name="entry"
    )

    expected_non_en_result = Series(
        data=[
            "",
            ["Ce ligne est Francais une", "Ce ligne est Francais deux", "This statement is English", "Ce ligne est Francais"]
        ],
        index=[0, 1], name="entry"
    )

    en_entries, non_en_entries = xmle.get_language_sections(entries)
    assert en_entries.equals(expected_en_result)
    assert non_en_entries.equals(expected_non_en_result)


def test_generate_xml(): # TODO update once generate_xml has been updated
    pass
    # in_xml = ET.parse("tests\\title_xml_example_1.xml")
    # root = in_xml.getroot()
    # all_lines = xmle.extract_lines(root)
    # title, all_title_indices, o_l = xmle.find_headings(all_lines)

    # out_xml = xmle.generate_xml(all_title_indices, all_lines, )

