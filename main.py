import glob
import json
import os

import pandas as pd

import emp.dataset as data

def create_title_loc_df():
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

def post_process_outputs(jsons, csv_out):
    header_template = pd.read_csv("data/external/Books_template.csv", nrows=2, encoding="utf8")
    jsons = glob.glob(jsons)
    json_dict = {os.path.basename(j).split(".")[0].replace("_", " ").title().replace("Al-", "al-"): json.load(open(j)) for j in jsons}
    metadata_df = data.process_output_to_csv(json_dict)
    marc_df = data.post_process_csv(metadata_df=metadata_df, header_template=header_template)
    marc_df.to_csv(csv_out, encoding="utf-8-sig")
    return title_loc_df

if __name__ == "__main__":
    trial_titles = pd.read_csv("data/external/trial_harry_jan_2026.csv", encoding="utf-8", header=None)
    trial_titles["title"] = trial_titles[1].apply(lambda x: x[:-5]).str.strip()

    title_loc_df = create_title_loc_df()
    trial_texts = title_loc_df.set_index("short_title_titles").loc[trial_titles["title"]]
    prompts = {title: data.gen_prompt(entry_text=text, book_title=title) for title, text in zip(trial_texts.index, trial_texts["entry_text"].values)}

    CREATE_PROMPTS = False
    CREATE_JSON = False

    if CREATE_PROMPTS:
        for title, prompt in prompts.items():
            with open(f"models/prompts/non_gt_prompts/batch_260130/{title.lower().replace(" ", "_")}.txt", "w", encoding="utf8") as f:
                f.write(prompt)

            if CREATE_JSON and not os.path.exists(f"data/processed/model_outputs/non_gt_outputs/batch_260130/{title.lower().replace(" ", "_")}.json"):
                with open(f"data/processed/model_outputs/non_gt_outputs/batch_260130/{title.lower().replace(" ", "_")}.json", "w") as f:
                    f.write("")

    post_process_outputs(jsons="data/processed/non_gt_outputs/batch_260130/*.json", csv_out="data/processed/non_gt_outputs/batch_260130/260130_postproc.csv")
    print("")
