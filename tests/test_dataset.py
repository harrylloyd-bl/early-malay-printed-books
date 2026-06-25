import glob
import json
import os
import emp.dataset as data
import pandas as pd
import pytest

proudfoot_text = data.parse_proudfoot("data/raw/emp.pdf")

@pytest.fixture
def text():
    return proudfoot_text

@pytest.fixture
def preprocessed_text(text):
    return data.preprocess_text(text)

@pytest.fixture
def all_works(preprocessed_text):
    text = preprocessed_text
    all_titles_raw = data.gen_title_lines(text)
    all_titles = data.manual_merge(all_titles_raw, "data/interim/lines_to_concatenate_with_text.txt")
    works = data.select_works(all_titles)
    return works

@pytest.fixture
def works(all_works):
    return data.gen_short_titles(all_works, data.shorten_proudfoot_title)

@pytest.fixture
def title_loc_df(works, preprocessed_text):
    desc_lines, _ = data.gen_desc_lines(preprocessed_text)
    title_loc_df = data.gen_title_loc_df(works["short_title_titles"], desc_lines)
    return title_loc_df

@pytest.fixture
def manual_check_df():
    manual_check_df = pd.read_csv("data/interim/missing_title_adjacent_manual_check.csv", encoding="UTF-8-SIG", index_col=1)
    return manual_check_df

def test_parse_proudfoot(text):
    
    assert len(text["desc"]) == 470
    assert len(text["titles"]) == 91
    
    assert text["desc"].get(125, None) is None
    assert text["desc"].get(596, None) is None

    assert text["desc"].get(126, "LOREM")[:5] == "EARLY"
    assert text["desc"].get(595, "IPSUM")[-5:] == "ly] \n"

    assert text["titles"].get(710, None) is None
    assert text["titles"].get(802, None) is None

    assert text["titles"].get(711, "LOREM")[:5] == "TITLE"
    assert text["titles"].get(801, "IPSUM")[-5:] == "773 \n"


def test_page_headers(text):
    even_headers = []
    for i in range(126, 596, 2):
        even_headers.append(text["desc"][i].split("\n")[0])

    odd_headers = []
    for i in range(127, 597, 2):
        odd_headers.append(text["desc"][i].split("\n")[0])

    assert len(even_headers) == 235
    assert len(odd_headers) == 235

    earl_headers = [h for h in even_headers if "EARL" in h]
    desc_headers = [h for h in odd_headers if "DESC" in h]

    assert len(earl_headers) == 231  # Manually checked
    assert len(desc_headers) == 234  # Manually checked


def test_preprocessed_text(preprocessed_text):
    text = preprocessed_text

    assert "\nTITI..ES " not in text["titles"][724]
    assert "\nTITLES " not in text["titles"][774]

    assert text["titles"][711][:10] == "TITLES \nAb"
    assert len(text["titles"][720]) == 1680
    assert len(text["titles"][721]) == 42
    assert len(text["titles"][722]) == 55


def test_preprocess_description_page():
    test_text = "DESC \nasdf \n;lkj\n40"
    processed = data.preprocess_description_page(test_text, '40')
    target = ["asdf", ";lkj"]
    assert processed == target


def test_description_ground_truth(preprocessed_text):
    text = preprocessed_text
    processed_desc_pages = []
    for i in range(126, 131):
        page_num = str(i - 28)
        page = text["desc"][i]
        processed_desc_pages.append(data.preprocess_description_page(page=page, page_num=page_num))
    
    for i, p in enumerate(processed_desc_pages[:5]):
        with open(f"data/processed/ground_truth/p{i+1}_column_parse.txt", encoding="utf8") as f:
            gt = [l.strip("\n") for l in f.readlines()]
            assert gt == p


def test_preprocess_titles_page(preprocessed_text):
    text = preprocessed_text

    earl_headers = [e for e in text["titles"].values() if "EARL" in e.split("\n")[0]]
    titles_headers = [e for e in text["titles"].values() if "LES" in e.split("\n")[0]]

    assert len(earl_headers) == 44
    assert len(titles_headers) == 41

    page_num_counts = [page.count(str(i - 28)) for i, page in text["titles"].items()]
    assert sum(page_num_counts) == 88  # Three removed by pre-processing

    with open("data/processed/ground_truth/all_titles_p683.txt", encoding="utf8") as f:
        all_titles_p683 = [x.strip("\n") for x in f.readlines()]
        all_titles_p683[4] = all_titles_p683[4].replace("\\", "")

    preproc_683 = data.preprocess_titles_page(text["titles"][711], 683)
    check_683 = [gt == preproc for gt, preproc in zip(all_titles_p683, preproc_683)]
    assert all(check_683)

    with open("data/processed/ground_truth/all_titles_p688.txt", encoding="utf8") as f:
        all_titles_p688 = [x.strip("\n") for x in f.readlines()]

    preproc_688 = data.preprocess_titles_page(text["titles"][716], 688)
    check_688 = [gt == preproc for gt, preproc in zip(all_titles_p688, preproc_688)]
    assert not all(check_688)  # This should fail until manual concatenation


def test_gen_title_lines(preprocessed_text):
    text = preprocessed_text
    title_lines = data.gen_title_lines(text)
    assert len(title_lines) == 4311


def test_gen_desc_lines(preprocessed_text):
    text = preprocessed_text
    desc_lines, line_page_lookup = data.gen_desc_lines(text)
    assert len(desc_lines) == 51208
    assert desc_lines[0] == "Abbas"
    assert desc_lines[-1] == "72,259-260,267-270,307-318 only]"

    assert len(line_page_lookup) == len(desc_lines)
    assert line_page_lookup[0] == 98
    assert line_page_lookup[len(line_page_lookup) - 1] == 567


def test_manual_merge(preprocessed_text):
    text = preprocessed_text
    all_titles = data.gen_title_lines(text)
    all_titles_corrected = data.manual_merge(all_titles, "data/interim/lines_to_concatenate_with_text.txt")
    assert len(all_titles_corrected) == 4174


def test_select_works(all_works):
    assert len(all_works) == 956
    
    with open("data/processed/ground_truth/28_main_titles.txt", encoding="utf8") as f:
        gt_main_works = [l.strip("\n") for l in f.readlines()]
    
    assert all([w == gt_w for w, gt_w in zip(all_works["title"], gt_main_works)])


def test_shorten_proudfoot_title():
    test_title = "short title a ±l984"
    target = "short title"
    assert data.shorten_proudfoot_title(test_title) == target

    test_title = "short title b ±l984"
    target = "short title b"
    assert data.shorten_proudfoot_title(test_title) == target


def test_shorten_aac_title():
    test_title = "short title a ±l984"
    target = "short title"
    assert data.shorten_aac_title(test_title) == target

    test_title = "short title b ±l984"
    target = "short title"
    assert data.shorten_aac_title(test_title) == target


def test_gen_short_titles(works):
    assert "short_title_titles" in works
    assert works["short_title_titles"].is_unique


def test_gen_aac_list():
    aac_file = "data/external/Proudfoot-BL collection-6.10.25.csv"
    aac_df = data.gen_aac_df(aac_file=aac_file)
    assert aac_df.shape == (686, 4)


@pytest.mark.for_review
def test_lookup_aac_titles(works):
    aac_file = "data/external/Proudfoot-BL collection-6.10.25.csv"
    aac_df = data.gen_aac_df(aac_file=aac_file)
    matched_works = data.lookup_aac_titles(aac_df=aac_df, works_df=works)
    assert len(matched_works[0]) == 2
    assert matched_works[0][0] == "? Bible: Mark"
    assert matched_works[-1][0] == "Šilam Bari"
    assert len(matched_works) == 374
    
    with open("data/processed/ground_truth/50_matched_works.txt", encoding="utf8") as f:
        gt_titles = [tuple(l.strip("\n").split(", ")) for l in f.readlines()]
    
    assert all([t in matched_works for t in gt_titles])


def test_find_nearest_line(title_loc_df, preprocessed_text):
    desc_lines, _ = data.gen_desc_lines(preprocessed_text)
    row = title_loc_df.iloc[1]
    res = data.find_nearest_line(row=row, desc_lines=desc_lines)
    target = ("Abdau", 100.0, 42)
    assert res == target

    row = title_loc_df.iloc[12]
    res = data.find_nearest_line(row=row, desc_lines=desc_lines)
    target = ("Adab ai-Fatly", 84.61538461538461, 1667)
    assert res == target


def test_gen_title_loc_df(title_loc_df):
    col_list = ['short_title_desc', 'entry_start', 'min_line', 'max_line']
    assert title_loc_df.columns.tolist() == col_list
    assert len(title_loc_df) == 949


def test_apply_find_nearest(title_loc_df, preprocessed_text):
    desc_lines, _ = data.gen_desc_lines(preprocessed_text)
    title_loc_df = data.apply_find_nearest(title_loc_df=title_loc_df, desc_lines=desc_lines)
    assert ["nearest_line", "similarity", "nearest_line_idx"] == title_loc_df.columns.tolist()[-3:]
    assert not title_loc_df.query("similarity >= 90")["short_title_desc"].dropna().hasnans
    assert not title_loc_df.query("similarity >= 90")["short_title_desc"].dropna().hasnans


def test_gen_manual_check_df(title_loc_df, preprocessed_text):
    _, line_page_lookup = data.gen_desc_lines(preprocessed_text)
    manual_check_df = data.gen_manual_check_df(title_loc_df=title_loc_df, line_page_lookup=line_page_lookup)
    assert "min_line_page" in manual_check_df
    assert len(manual_check_df) == 848


def test_extract_clean_entries(manual_check_df, title_loc_df, preprocessed_text):
    desc_lines, _ = data.gen_desc_lines(preprocessed_text)
    title_loc_df = data.apply_find_nearest(title_loc_df, desc_lines)
    title_loc_df = data.extract_clean_entries(manual_check_df=manual_check_df, title_loc_df=title_loc_df, desc_lines=desc_lines)
    
    assert "correct_title" in title_loc_df
    assert "entry_text" in title_loc_df

    assert title_loc_df.loc[title_loc_df.query("short_title_desc == 'Akidat al-Munjian'").index, "entry_start"].values[0] == 2457
    assert title_loc_df.iloc[-1, -3] == 51208

    assert not title_loc_df["entry_start"].hasnans
    assert not title_loc_df["entry_end"].hasnans

    # Check that all lines in the Description section are covered by the catalogue entries
    entry_lines_set = set()
    for row in title_loc_df.iterrows():
        s = set(range(row[1]["entry_start"], row[1]["entry_end"] + 1))
        entry_lines_set |= s

    assert len(set(range(0, 51207)) - entry_lines_set) == 0 

    # check the first five entries have been picked up correctly
    entry_gt = {
        "Abbas": desc_lines[0:42],
        "Abdau": desc_lines[42:91],
        "Abdullah": desc_lines[91:572],
        "Abdullah dan Sa bat": desc_lines[572:677],
        "Abdul Muluk": desc_lines[677:1091]
    }

    for gt_title, gt_text in entry_gt.items():
        entry_text = title_loc_df.query(f"correct_title == '{gt_title}'").loc[:, "entry_text"].values[0].split("\n")
        assert gt_text == entry_text


def test_extract_bl_shelfmark():
    test_sm = "BL 00000.a.11"
    res = data.extract_bl_shelfmark(test_sm)
    target = "00000.a.11"
    assert res == target

    test_sm = "BL 00000.a.11; Barbican 4001"
    res = data.extract_bl_shelfmark(test_sm)
    target = "00000.a.11"
    assert res == target

    test_sm = "BL 0000l.1.l1; Barbican 4001"
    res = data.extract_bl_shelfmark(test_sm)
    target = "00001.l.11"
    assert res == target


def test_process_output_to_csv():
    jsons = glob.glob("data/processed/batch_251219_ground_truth/*.json")
    json_dict = {os.path.basename(j).split(".")[0].replace("_", " ").title(): json.load(open(j)) for j in jsons}
    metadata_df = data.process_output_to_csv(json_dict)
    assert metadata_df.shape == (50, 13)
    assert metadata_df.columns.to_list() == ['shelfmark', 'date_1', 'name', 'title', 'place_of_publication',
       'publisher', 'date_of_publication_in_arabic_or_roman_numerals',
       'extent', 'dimensions', 'general_notes', 'citation_ref_note',
       'method_of_acquisition', 'unclassified_text']


def test_post_process_csv():
    header_template = pd.read_csv("data/external/Books_template.csv", nrows=2, encoding="utf8")
    jsons = glob.glob("data/processed/batch_251219_ground_truth/*.json")
    json_dict = {os.path.basename(j).split(".")[0].replace("_", " ").title(): json.load(open(j)) for j in jsons}
    metadata_df = data.process_output_to_csv(json_dict)
    marc_df = data.post_process_csv(metadata_df=metadata_df, header_template=header_template)
    assert marc_df.shape == (52, 94)
