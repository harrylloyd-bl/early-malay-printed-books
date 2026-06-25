from typing import Any
import asyncio
from datetime import datetime
import glob
import json
import logging
import os

import pandas as pd

import emp.dataset as data
from emp.config import *


def post_process_outputs(text_outputs, csv_out):
    header_template = pd.read_csv("data/external/Books_template.csv", nrows=2, encoding="utf8")
    outputs = glob.glob(text_outputs)
    json_dict: dict[str, str | dict[str, list[dict[str, str]]]] = {}
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
    marc_df.to_excel(csv_out.replace(".csv", ".xlsx"))


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f"logs/{datetime.now().strftime("%Y%m%d_%H%M")}_main.log", encoding="utf8", level=logging.DEBUG)

    CREATE_TITLE_LOC_DF = False

    USE_DEFAULT_WORKSPACE = True
    CALL_API = False
    CREATE_PROMPTS = False
    CREATE_JSON = False
    
    POST_PROCESS = True

    BATCH = "260623"
    MODEL = "qwen3.6-35b-a3b"
    
    if not os.path.exists(os.path.join(DATA_DIR, f"processed/batch_{BATCH}/")):
        os.mkdir(os.path.join(DATA_DIR, f"processed/batch_{BATCH}/"))

    trial_loc_path = os.path.join(DATA_DIR, "interim/title_loc_df.csv")
    if CREATE_TITLE_LOC_DF:
        title_loc_df = data.create_title_loc_df()
        title_loc_df.to_csv(trial_loc_path, encoding="utf-8-sig", index=True)
        title_loc_df.to_excel(trial_loc_path.replace("csv", "xlsx"), index=True)
    else:
        title_loc_df = pd.read_csv(trial_loc_path, encoding="utf-8-sig", index_col=0)

    prepared_entry_df = pd.read_csv(os.path.join(DATA_DIR, "interim/title_dedup_boll_emp.csv"), encoding="utf-8-sig").set_index("title")
    entries = {title: data.gen_prompt(entry_text=row["entry_text"], book_title=title) for title, row in prepared_entry_df.iloc[5:].iterrows()}

    if USE_DEFAULT_WORKSPACE:
        base_url=f"https://{os.environ['SINGAPORE_API_HOST']}/compatible-mode/v1"
    else:
        base_url=f"https://{os.environ['EU_API_HOST']}/compatible-mode/v1"

    if CALL_API:
        print("Running event loop...")
        # structured_entries = asyncio.run(data.structure_all_entries(base_url=base_url, entries=entries, max_concurrent=15, model=MODEL, logger=logger))
        for res in structured_entries:
            with open(f"data/processed/batch_{BATCH}/{res[0].lower().replace(" ", "_").replace(":", "_")}.txt", "w") as f:
                logging.info(f"Output token count: {len(res[1].choices[0].message.content.split(" "))}")
                f.write(res[1].choices[0].message.content.strip("```").strip("json"))

    if CREATE_PROMPTS:
        for title, prompt in entries.items():
            with open(f"models/prompts/non_gt_prompts/batch_{BATCH}/{title.lower().replace(" ", "_")}.txt", "w", encoding="utf8") as f:
                f.write(prompt)

            json_path = os.path.join(DATA_DIR, f"processed/batch_260130/{title.lower().replace(" ", "_")}.json")
            if CREATE_JSON and not os.path.exists(json_path):
                with open(json_path, "w") as f:
                    f.write("")

    if POST_PROCESS:
        post_process_outputs(text_outputs=os.path.join(DATA_DIR, f"processed/batch_{BATCH}/*.txt"), csv_out=os.path.join(DATA_DIR, f"processed/batch_{BATCH}/{BATCH}_postproc.csv"))
    print("")
