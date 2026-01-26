import pandas as pd

import emp.dataset as data

def main():
    text = data.parse_proudfoot("data/raw/emp.pdf")
    preproc_text = data.preprocess_text(text)
    all_titles_raw = data.gen_title_lines(preproc_text)
    all_titles = data.manual_merge(all_titles=all_titles_raw, merge_file="data/processed/lines_to_concatenate_with_text.txt")
    works = data.select_works(all_titles)
    works = data.gen_short_titles(works, data.shorten_proudfoot_title)

    desc_lines, _ = data.gen_desc_lines(preproc_text)
    title_loc_df = data.gen_title_loc_df(works=works["short_title_titles"], desc_lines=desc_lines)
    title_loc_df = data.apply_find_nearest(title_loc_df, desc_lines)

    manual_check_df = pd.read_csv("data/interim/missing_title_adjacent_manual_check.csv", encoding="UTF8", index_col=0)
    title_loc_df = data.extract_clean_entries(manual_check_df, title_loc_df, desc_lines)

    return title_loc_df

if __name__ == "__main__":
    title_loc_df = main()
    print(title_loc_df.shape)
