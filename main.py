import asyncio
from datetime import datetime
import glob
import json
import logging
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

def post_process_outputs(text_outputs, csv_out):
    header_template = pd.read_csv("data/external/Books_template.csv", nrows=2, encoding="utf8")
    outputs = glob.glob(text_outputs)
    json_dict = {}
    for f in outputs:
        try:
            short_name = os.path.basename(f).split(".")[0].replace("_", " ").title().replace("Al-", "al-")
            work_json = json.loads(open(f).read().strip("````").strip("json"))
            json_dict[short_name] = work_json
        except json.decoder.JSONDecodeError:
            print(str(res[1].choices[0].finish_reason) + "\n")
            print(str(res[1].choices[0].message.refusal) + "\n")
            print(str(res[1].choices[0].message.content))
            json_dict[short_name] = "JSON DECODE FAILURE"
    
    metadata_df = data.process_output_to_csv(json_dict)
    marc_df = data.post_process_csv(metadata_df=metadata_df, header_template=header_template)
    marc_df.to_csv(csv_out, encoding="utf-8-sig")
    return title_loc_df

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f"logs/{datetime.now().strftime("%Y%m%d_%H%M")}_main.log", encoding="utf8", level=logging.DEBUG)
    trial_titles = pd.read_csv("data/external/trial_harry_jan_2026.csv", encoding="utf-8", header=None)
    trial_titles["title"] = trial_titles[1].apply(lambda x: x[:-5]).str.strip()

    title_loc_df = create_title_loc_df()
    trial_texts = title_loc_df.set_index("short_title_titles").loc[trial_titles["title"]]
    entries = {title: data.gen_prompt(entry_text=text, book_title=title) for title, text in zip(trial_texts.index, trial_texts["entry_text"].values)}

    CALL_API = False
    CREATE_PROMPTS = False
    CREATE_JSON = False
    POST_PROCESS = True
    BATCH = "260317"
    MODEL = "qwen3.5-35b-a3b"
    
    if not os.path.exists(f"data/processed/batch_{BATCH}/"):
        os.mkdir(f"data/processed/batch_{BATCH}/")

    if CALL_API:
        print("Running event loop...")
        structured_entries = asyncio.run(data.structure_all_entries(entries=entries, max_concurrent=15, model=MODEL, logger=logger))
        for res in structured_entries:
            with open(f"data/processed/batch_{BATCH}/{res[0].lower().replace(" ", "_")}.txt", "w") as f:
                logging.info(f"Output token count: {len(res[1].choices[0].message.content.split(" "))}")
                f.write(res[1].choices[0].message.content.strip("```").strip("json"))


    if CREATE_PROMPTS:
        for title, prompt in entries.items():
            with open(f"models/prompts/non_gt_prompts/batch_{BATCH}/{title.lower().replace(" ", "_")}.txt", "w", encoding="utf8") as f:
                f.write(prompt)

            if CREATE_JSON and not os.path.exists(f"data/processed/batch_260130/{title.lower().replace(" ", "_")}.json"):
                with open(f"data/processed/batch_{BATCH}/{title.lower().replace(" ", "_")}.json", "w") as f:
                    f.write("")

    if POST_PROCESS:
        post_process_outputs(text_outputs=f"data/processed/batch_{BATCH}/*.txt", csv_out=f"data/processed/batch_{BATCH}/{BATCH}_postproc.csv")
    print("")
